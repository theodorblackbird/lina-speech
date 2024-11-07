from __future__ import annotations
import torch
import torch.nn as nn
import os

import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from collections import defaultdict

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttention, BlindCrossAttention, CrossAttentionPP
from model.base_blocks import MixingBlock, SwiGLU

from typing import Optional, Tuple, List

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla
from fla.ops.simple_gla import chunk_simple_gla
from fla.models.utils import Cache


if "GRAD_CKPT" in os.environ:
    def maybe_grad_ckpt(f):
        def grad_ckpt_f(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(f, *args, **kwargs, use_reentrant=False)
        return grad_ckpt_f
else:
    def maybe_grad_ckpt(f):
        return f
# pyright: basic
#############
# this part is adapted from https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/gla.py#L20

# -*- coding: utf-8 -*-

# "Gated Linear Attention Transformers with Hardware-Efficient Training"[https://arxiv.org/abs/2312.06635]



class GatedLinearAttention(nn.Module):

    def __init__(
            self,
            mode: str = 'fused_chunk',
            hidden_size: int = 1024,
            expand_k: float = 1.0,
            expand_v: float = 2.0,
            num_heads: int = 4,
            use_short_conv: bool = False,
            conv_size: int = 4,
            conv_bias: bool = False,
            share_conv_kernel: bool = False,
            gate_fn: str = 'swish',
            layernorm_eps: float = 1e-5,
            gate_logit_normalizer: int = 16,
            gate_low_rank_dim: int = 16,
            clamp_min: Optional[float] = None,
            fuse_norm: bool = True,
            layer_idx: int = None,
            **kwargs
            ) -> GatedLinearAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk', 'naive'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        #self.gk_proj = nn.Linear(hidden_size, self.num_heads)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        if gate_fn == 'swish' and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, eps=layernorm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            reset_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            reset_val: float = -20,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            **kwargs
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                conv_state_q = last_state[0] if use_cache else None
                conv_state_k = last_state[1] if use_cache else None
                conv_state_v = last_state[2] if use_cache else None
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                q = self.q_conv1d(q, attention_mask, conv_state_q)
                k = self.k_conv1d(k, attention_mask, conv_state_k)
                v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))
        gk = rearrange(self.gk_proj(hidden_states), 'b n (h d) -> b h n d', h=self.num_heads)
        #gk = rearrange(self.gk_proj(hidden_states), 'b l h -> b h l')
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer


        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        if reset_mask is not None:
            gk = gk.masked_fill(reset_mask.unsqueeze(1).unsqueeze(3), reset_val)
            #gk = gk.masked_fill(reset_mask, -1e38)
        
        recurrent_state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'inference':
            o, recurrent_state = fused_recurrent_gla(q, k, v, gk, initial_state=self.state, output_final_state=True)
            self.state = recurrent_state
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'naive':
            o, recurrent_state = naive_recurrent_gla(q, k, v, gk, initial_state=self.state, output_final_state=use_cache)
        elif mode == 'init_state_tuning':
            bs = q.shape[0]
            initial_state=repeat(self.state, "1 ... -> bs ...", bs=bs)
            o, recurrent_state = fused_recurrent_gla(q, k, v, gk, initial_state=initial_state, output_final_state=True)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None and not self.training:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state, recurrent_state)
                else:
                    last_state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(hidden_states)
        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = self.g_norm(o)
            o = rearrange(o, 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size

########


class AttentiveGLA(AttentiveRNN):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            heads: int,
            dropout_att: float=0.0,
            dropout: float=0.,
            d_blind: int=None,
            blind: bool =False,
            cross_att_pp: bool=False,
            rotary: bool =False,
            use_short_conv: bool=False,
            expand_k: float=1.,
            expand_v: float=2.,
            pos_type="sinusoidal",
            ):
        super().__init__()
        block = lambda d, h, i: MixingBlock(lambda: GatedLinearAttention(hidden_size=d, num_heads=h, use_short_conv=use_short_conv, expand_k=expand_k, expand_v=expand_v, layer_idx=i),
                                         lambda: SwiGLU(d),
                                         lambda: nn.LayerNorm(d),
                                         dropout=dropout,
                                         )
        self.encoder = nn.ModuleList([block(d_model, heads, i) for i in range(n_layer)])
        self.decoder = nn.ModuleList([block(d_model, heads, i) for i in range(n_layer, 2*n_layer)])
        if d_blind is None:
            d_blind=d_model

        if blind:
            self.cross_att = BlindCrossAttention(d_model, d_model, d_model, 1, block(d_blind, heads, 2*n_layer), dropout_att, pos_dim=d_blind, rotary=rotary, pos_type=pos_type)
        elif cross_att_pp:
            self.cross_att = CrossAttentionPP(d_model, block(d_model, heads), 1,ca_dropout=dropout_att)
        else:
            self.cross_att = CrossAttention(d_model, d_model, d_model, heads, dropout_att)

    def forward(self, x, ctx, mask=None, pos=None, reset_mask=None, attention_only=None, forced_attention=None, init_state=None, crossatt_pos=None):
        
        for e in self.encoder:
            if self.training:
                e = maybe_grad_ckpt(e)
            x = e(x, use_cache=init_state is not None, past_key_values=init_state)

        v, att = self.cross_att(x, ctx, mask=mask, reset_mask=reset_mask, pos=crossatt_pos)
        x = x + v
        for d in self.decoder:
            if self.training:
                d = maybe_grad_ckpt(d)
            x = d(x, use_cache=init_state is not None, past_key_values=init_state)
        return x, att

    def init_state(self, max_seqlen=1000, batch_size=16):
        cache = Cache()
        for i, e in enumerate(self.encoder):
            cache.update(e.tmix.init_state(batch_size), i, offset=0)
            
        for i, d in enumerate(self.decoder):
            cache.update(d.tmix.init_state(batch_size), i + len(self.encoder), offset=0)

        if hasattr(self.cross_att, "pos_net"):
            cache.update(d.tmix.init_state(batch_size), len(self.decoder) + len(self.encoder), offset=0)
        
        return cache

    def get_state_from_params(self, params, batch_size, scale=0.02):
        cache = self.init_state(batch_size=batch_size)
        for i, x in enumerate(params):
            if len(x) == 2:
                state = einsum(*x, "b r h k vv, b r h kk v -> b h k v")*scale
            else:
                state = x[0]
            state = repeat(state, "1 ... -> bs ...", bs=batch_size).clone()
            cache.states[i] = cache.states[i][:-1] + (state,)
            
        return cache

    def to_mode(self, mode):
        for b in self.encoder:
            b.tmix.mode = mode
        for b in self.decoder:
            b.tmix.mode = mode
        if hasattr(self.cross_att, "pos_net"):
            self.cross_att.pos_net.mode = mode
        

    def get_init_state_tuning_params(self, lora: Optional[int]=None, scale: float=0.02, device=None):
        parameters = []
        def produce_params(h, k, v, lora:Optional[int]=None):
            if lora is not None:
                k = nn.Parameter(torch.randn(1, lora, h, k, 1, device=device))
                v = nn.Parameter(torch.randn(1, lora, h, 1, v, device=device)*scale)
                return k, v
            else:
                return nn.Parameter(torch.randn(1, h, k, v, device=device)*scale)

        for b in self.encoder:
            b = b.tmix
            k, v, h = b.head_qk_dim, b.head_v_dim, b.num_heads
            parameters.append(produce_params(h, k, v, lora=lora))

        for b in self.decoder:
            b = b.tmix
            k, v, h = b.head_qk_dim, b.head_v_dim, b.num_heads
            parameters.append(produce_params(h, k, v, lora=lora))
        
        return parameters

    def step(self, y_embd, x_enc, time_step, cache):
        for e in self.encoder:
            y_embd = e(y_embd, past_key_values=cache, use_cache=True)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step, past_key_values=cache, use_cache=True)
        y_embd = y_embd + v
        for d in self.decoder:
            y_embd = d(y_embd, past_key_values=cache, use_cache=True)
        return y_embd, att, cache

class CrossAttGLA(AttentiveRNN):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            cross_att_layers: list[int],
            heads: int,
            cross_att_heads: int,
            dropout_att: float=0.0,
            dropout: float=0.,
            rotary: bool =False,
            use_short_conv: bool=False,
            expand_k: float=1.,
            expand_v: float=2.,

            ):
        super().__init__()
        block = lambda d, h, i: MixingBlock(lambda: GatedLinearAttention(hidden_size=d, num_heads=h, use_short_conv=use_short_conv, expand_k=expand_k, expand_v=expand_v, layer_idx=i),
                                         lambda: SwiGLU(d),
                                         lambda: nn.LayerNorm(d),
                                         dropout=dropout,
                                         )

        self.blocks = nn.ModuleList([block(d_model, heads, i) for i in range(n_layer)])
        self.cross_att = nn.ModuleList([CrossAttention(d_model, d_model, d_model, cross_att_heads, dropout=dropout_att, rotary=rotary) for _ in cross_att_layers])
        self.cross_att_layers = cross_att_layers

    def forward(self, x, ctx, mask=None, pos=None, reset_mask=None, attention_only=None, forced_attention=None, init_state=None, **kwargs):
        y = x
        idx = {k: v for v, k in enumerate(self.cross_att_layers)}
        att = None
        for i, b in enumerate(self.blocks):
            if self.training:
                b = maybe_grad_ckpt(b)
            y = b(y)
            if i in self.cross_att_layers:
                v, att = self.cross_att[idx[i]](y, ctx, mask=mask.unsqueeze(1))
                y = y + v
        return y, att

    def init_state(self, max_seqlen=1000):
        for b in self.blocks:
            b.tmix.state = None
            b.tmix.mode = "inference"

    def step(self, y, ctx, time_step):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.cross_att_layers[0]](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1)

class CrossAttGLAV2(AttentiveRNN):
    def __init__(
            self,
            n_layer,
            crossatt_layers: List[int],
            d_model,
            n_heads,
            crossatt_heads: int = 2,
            dropout_att: float = 0.1,
            rotary: bool = False,
            ):
        super().__init__()


        block = lambda: MixingBlock(lambda: GatedLinearAttention(hidden_size=d_model, num_heads=n_heads),
                                         lambda: SwiGLU(d_model),
                                         lambda: nn.LayerNorm(d_model),
                                         )
        self.blocks = nn.ModuleList([block() for i in range(n_layer)])
        self.cross_att = nn.ModuleDict({str(i): CrossAttention(d_model, d_model, d_model, crossatt_heads, dropout=dropout_att, rotary=rotary) for i in crossatt_layers})
        self.crossatt_layers = crossatt_layers
        self.n_layer = n_layer

    def forward(self, x, ctx, mask=None, pos=None, **kwargs):
        y = x
        att = None
        for i, b in enumerate(self.blocks):
            y = torch.utils.checkpoint.checkpoint(b, y, use_reentrant=False)
            if i in self.crossatt_layers:
                v, _att = self.cross_att[str(i)](y, ctx, mask=mask.unsqueeze(1))
                y = y + v
                if _att is not None:
                    if att is not None:
                        att = torch.cat((att, _att), dim=1)
                    else:
                        att = _att
        return y, att

    def init_state(self, max_seqlen=1000, state=None):
        if state is None:
            state = defaultdict(lambda: None)
        for i, b in enumerate(self.blocks):
            b.tmix.state = state[i]
            b.tmix.mode = "inference"



    def step(self, y, ctx, time_step):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y)
            if i in self.crossatt_layers:
                v, att = self.cross_att[str(i)](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1)

