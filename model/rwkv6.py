from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttention, BlindCrossAttention, CrossAttentionPP
from model.base_blocks import MixingBlock, SwiGLU
from fla.layers.rwkv6 import RWKV6Attention

from typing import Optional, Tuple


class AttentiveRWKV6(AttentiveRNN):
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
            ):
        super().__init__()
        block = lambda d, h: MixingBlock(lambda: RWKV6Attention(hidden_size=d, num_heads=h),
                                         lambda: SwiGLU(d),
                                         lambda: nn.LayerNorm(d),
                                         dropout=dropout,
                                         )
        self.encoder = nn.ModuleList([block(d_model, heads) for i in range(n_layer)])
        self.decoder = nn.ModuleList([block(d_model, heads) for i in range(n_layer)])
        if d_blind is None:
            d_blind=d_model

        if blind:
            self.cross_att = BlindCrossAttention(d_model, d_model, d_model, 1, block(d_blind, heads), dropout_att, pos_dim=d_blind, rotary=rotary)
        elif cross_att_pp:
            self.cross_att = CrossAttentionPP(d_model, block(d_model, heads), 1,ca_dropout=dropout_att)
        else:
            self.cross_att = CrossAttention(d_model, d_model, d_model, heads, dropout_att)

    def forward(self, x, ctx, mask=None, pos=None, reset_mask=None):
        for e in self.encoder:
            x = torch.utils.checkpoint.checkpoint(e, x, reset_mask=reset_mask, use_reentrant=False)
        v, att = self.cross_att(x, ctx, mask=mask, pos=pos, reset_mask=reset_mask)
        x = x + v
        for d in self.decoder:
            x = torch.utils.checkpoint.checkpoint(d, x, reset_mask=reset_mask, use_reentrant=False)
        return x, att

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
        for e in self.encoder:
            y_embd = e(y_embd)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step)
        y_embd = y_embd + v
        for d in self.decoder:
            y_embd = d(y_embd)
        return y_embd, att

class CrossAttGLA(AttentiveRNN):
    def __init__(
            self,
            d_model,
            n_layer,
            heads=4,
            crossatt_heads=4,
            n_crossatt_layer=2,
            dropout_att=0.1,
            rotary=False,
            ):
        super().__init__()
        block = lambda d, h: MixingBlock(lambda: GatedLinearAttention(hidden_size=d, num_heads=h),
                                         lambda: SwiGLU(d),
                                         lambda: nn.LayerNorm(d),
                                         )
        self.blocks = nn.ModuleList([block(d_model, heads) for i in range(n_layer)])
        self.cross_att = nn.ModuleList([CrossAttention(d_model, d_model, d_model, crossatt_heads, dropout=dropout_att, rotary=rotary) for i in range(2)])
        self.cross_att_layers = list(range(n_layer//2-(n_crossatt_layer//2), n_layer//2 + (n_crossatt_layer//2) + int(n_crossatt_layer%2)))
        print(self.cross_att_layers)
        self.n_layer = n_layer

    def forward(self, x, ctx, mask=None, pos=None):
        y = x
        att = None
        for i, b in enumerate(self.blocks):
            y = torch.utils.checkpoint.checkpoint(b, y, use_reentrant=False)
            #y = b(y)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.cross_att_layers[0]](y, ctx, mask=mask.unsqueeze(1))
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
