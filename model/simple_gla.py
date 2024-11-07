from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import os
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel

from fla.layers.simple_gla import SimpleGatedLinearAttention
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, RMSNorm
from fla.modules.activations import swiglu_linear

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttention, BlindCrossAttention, CrossAttentionPP
from model.base_blocks import MixingBlock, SwiGLU

from einops import rearrange, repeat
from collections import defaultdict


if "GRAD_CKPT" in os.environ:
    def grad_ckpt(f):
        def grad_ckpt_f(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(f, *args, **kwargs, use_reentrant=False)
        return grad_ckpt_f
else:
    def grad_ckpt(f):
        return f

class GLAMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> GLAMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False).bfloat16()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        out = swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
        return  out


class GLABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, layer_idx: int, use_short_conv: bool = False, attn_mode: str = "chunk"):
        super().__init__()
        self.hidden_size = hidden_size

        self.attn_norm = RMSNorm(hidden_size=hidden_size)
        self.attn = SimpleGatedLinearAttention(
            mode=attn_mode,
            num_heads=num_heads,
            hidden_size=hidden_size,
            use_short_conv=use_short_conv,
            layer_idx=layer_idx
        )
        self.mlp_norm = RMSNorm(hidden_size=hidden_size)
        self.mlp = SwiGLU(hidden_size)
        #GLAMLP(
        #    hidden_size=hidden_size,
        #)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs

class AttentiveSimpleGLA(AttentiveRNN):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            heads: int,
            dropout_att: float=0.0,
            d_blind: int=None,
            blind: bool =False,
            cross_att_pp: bool=False,
            rotary: bool =False,
            use_short_conv: bool=False,
            pos_type="sinusoidal",
            dropout: float=0.,
            ):
        super().__init__()
        #block = lambda d, h, i: GLABlock(d, h, i, use_short_conv=use_short_conv)

        block = lambda d, h, i: MixingBlock(lambda: SimpleGatedLinearAttention(hidden_size=d, num_heads=h, use_short_conv=use_short_conv, layer_idx=i),
                                         lambda: SwiGLU(d),
                                         lambda: nn.LayerNorm(d),
                                         dropout=dropout,)
        self.encoder = nn.ModuleList([block(d_model, heads, i) for i in range(n_layer)])
        self.decoder = nn.ModuleList([block(d_model, heads, i + n_layer + 1) for i in range(n_layer)])

        if d_blind is None:
            d_blind=d_model

        if blind:
            self.cross_att = BlindCrossAttention(d_model, d_model, d_model, 1, block(d_blind, heads, n_layer), dropout_att, pos_dim=d_blind, rotary=rotary, pos_type=pos_type)
        elif cross_att_pp:
            self.cross_att = CrossAttentionPP(d_model, block(d_model, heads), 1,ca_dropout=dropout_att)
        else:
            self.cross_att = CrossAttention(d_model, d_model, d_model, heads, dropout_att)

    def forward(self, x, ctx, mask=None, pos=None, reset_mask=None, forced_attention=None, attention_only=None):
        for e in self.encoder:
            if self.training:
                e = grad_ckpt(e)
            x = e(x)

        v, att = self.cross_att(x, ctx, mask=mask, pos=pos, reset_mask=reset_mask)
        x = x + v
        
        for d in self.decoder:
            if self.training:
                d = grad_ckpt(d)
            x = d(x)
        return x, att

    def init_state(self, max_seqlen=1000, state=None):
        pass

    def get_state(self):
        pass

    def step(self, y_embd, x_enc, time_step):
        for e in self.encoder:
            y_embd, _, past_key_values = e(y_embd)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step)
        y_embd = y_embd + v
        for d in self.decoder:
            y_embd, _, past_key_values = d(y_embd)
        return y_embd, att


