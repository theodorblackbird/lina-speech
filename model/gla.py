from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttention, BlindCrossAttention
from model.base_blocks import MixingBlock, SwiGLU

from typing import Optional, Tuple

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


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
        share_conv_kernel: bool = True,
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

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
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
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        #if self.use_short_conv:
        if False:
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
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

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
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
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

def exists(x):
    return x is not None



class AttentiveGLA(AttentiveRNN):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_context: int,
        heads: int,
        dropout_att: float=0.0,
        dropout: float=0.,
        d_blind: int=None,
        blind: bool =False,
    ):
        super().__init__()
        if d_blind is None:
            d_blind=d_model
        block = lambda d, h: MixingBlock(lambda: GatedLinearAttention(hidden_size=d, num_heads=h),
                                                   lambda: SwiGLU(d),
                                                   lambda: nn.LayerNorm(d),
                                                    dropout=dropout,
                                                   )
        self.encoder = nn.Sequential(*[block(d_model, heads) for i in range(n_layer)])
        self.decoder = nn.Sequential(*[block(d_model, heads) for i in range(n_layer)])
        self.cross_att = (
            BlindCrossAttention(
                d_model, d_context, d_model, 1, block(d_blind, heads), dropout_att, pos_dim=d_blind
            )
            if blind
            else CrossAttention(d_model, d_context, d_model, heads, dropout_att)
        )

    def forward(self, x, ctx, x_mask=None, ctx_mask=None):
        if exists(x_mask) and exists(ctx_mask):
            mask = rearrange(x_mask, "b i -> b i 1") * rearrange(
                ctx_mask, "b j -> b 1 j"
            )
            mask = rearrange(mask, "b i j -> b 1 i j")
        else:
            mask = None

        y = torch.utils.checkpoint.checkpoint_sequential(self.encoder, len(self.encoder), x, use_reentrant=False)
        #y = self.encoder(x)
        v, att = self.cross_att(y, ctx, mask=mask)
        y = torch.utils.checkpoint.checkpoint_sequential(self.decoder, len(self.decoder), y + v, use_reentrant=False)
        #y = self.decoder(y + v)

        return y, att

    def init_state(self, max_seqlen=1000):
        for be, bd in zip(self.encoder, self.decoder):
            be.tmix.mode, bd.tmix.mode = 'inference', 'inference'
            be.tmix.state, bd.tmix.state = None, None
            be.tmix.use_short_conv = False
            bd.tmix.use_short_conv = False
        self.cross_att.pos_net.tmix.mode = 'inference'
        self.cross_att.pos_net.tmix.state = None
        self.cross_att.pos_net.tmix.use_short_conv = False

    def step(self, y_embd, x_enc, time_step):
        y_embd = self.encoder(y_embd)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step)
        y_embd = y_embd + v
        y_embd = self.decoder(y_embd)
        return y_embd, att

class CrossAttGLA(AttentiveRNN):
    def __init__(
        self,
        d_model,
        n_layer,
        d_context,
        heads,
        dropout_att=0.0,
        blind=False,
    ):
        super().__init__()
        block = lambda d, h: MixingBlock(lambda: GatedLinearAttention(d_model=d, num_head=h),
                                                   lambda: SwiGLU(d),
                                                   lambda: nn.LayerNorm(d),
                                                   )
        self.blocks = nn.ModuleList([block(d_model, heads) for i in range(n_layer)])
        self.cross_att = nn.ModuleList([CrossAttention(d_model, d_context, d_model, heads, dropout_att) for i in range(2)])
        self.cross_att_layers = list(range(n_layer//2, n_layer//2 + 2))
        self.n_layer = n_layer

    def forward(self, x, ctx, x_mask=None, ctx_mask=None):
        if exists(x_mask) and exists(ctx_mask):
            mask = rearrange(x_mask, "b i -> b i 1") * rearrange(
                ctx_mask, "b j -> b 1 j"
            )
            mask = rearrange(mask, "b i j -> b 1 i j")
        else:
            mask = None
        y = x
        for i, b in enumerate(self.blocks):
            y = torch.utils.checkpoint.checkpoint(b, y, use_reentrant=False)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.n_layer//2](y, ctx, mask=mask)
                y = y + v

        return y, att

    def init_state(self, max_seqlen=1000):
        for b in self.blocks:
            b.att.state = None
            b.att.mode = "inference"
#            be.att.state, bd.att.state = None, None
#        self.cross_att.pos_net.att.mode = 'inference'
#        self.cross_att.pos_net.att.state = None
#
    def step(self, y, ctx, time_step):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.n_layer//2](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1)

class MixedAttGLA(AttentiveRNN):
    def __init__(
        self,
        d_model,
        n_layer,
        d_context,
        heads,
        d_blind=64,
        dropout_att=0.0,
    ):
        super().__init__()
        block = lambda d, h: MixingBlock(lambda: GatedLinearAttention(d_model=d, num_head=h),
                                                   lambda: SwiGLU(d),
                                                   lambda: nn.LayerNorm(d),
                                                   )
        self.blocks = nn.ModuleList([block(d_model, heads) for i in range(n_layer)])
        self.cross_att = nn.ModuleList([
            CrossAttention(d_model, d_context, d_model, heads, dropout_att),
            BlindCrossAttention(d_model, d_context, d_model, 1, block(d_blind, 1), dropout_att),
            ])
        self.cross_att_layers = list(range(n_layer//2, n_layer//2 + 2))
        self.n_layer = n_layer

    def forward(self, x, ctx, x_mask=None, ctx_mask=None):
        if exists(x_mask) and exists(ctx_mask):
            mask = rearrange(x_mask, "b i -> b i 1") * rearrange(
                ctx_mask, "b j -> b 1 j"
            )
            mask = rearrange(mask, "b i j -> b 1 i j")
        else:
            mask = None
        y = x
        for i, b in enumerate(self.blocks):
            y = torch.utils.checkpoint.checkpoint(b, y, use_reentrant=False)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.n_layer//2](y, ctx, mask=mask)
                y = y + v

        return y, att

    def init_state(self, max_seqlen=1000):
        for b in self.blocks:
            b.att.state = None
            b.att.mode = "inference"
#            be.att.state, bd.att.state = None, None
        self.cross_att[1].pos_net.att.mode = 'inference'
        self.cross_att[1].pos_net.att.state = None
#
    def step(self, y, ctx, time_step):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.n_layer//2](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1)

