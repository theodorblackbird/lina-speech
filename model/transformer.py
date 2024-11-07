from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttention, BlindCrossAttention
from model.base_blocks import MixingBlock, SwiGLU, SelfAttention

from typing import Optional, Tuple
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache



class TransformerCrossAtt(AttentiveRNN):
    def __init__(
            self,
            n_layer,
            crossatt_layers,
            d_model,
            n_heads,
            crossatt_heads: int = 2,
            dropout_att: float = 0.1,
            crossatt_rotary: bool = True,
            ):
        super().__init__()


        block = lambda: MixingBlock(lambda: SelfAttention(d_model, n_heads, is_causal=True, rotary=True),
                                         lambda: SwiGLU(d_model),
                                         lambda: nn.LayerNorm(d_model),
                                         )
        self.blocks = nn.ModuleList([block() for i in range(n_layer)])
        self.cross_att = nn.ModuleDict({str(i): CrossAttention(d_model, d_model, d_model, crossatt_heads, dropout=dropout_att, rotary=crossatt_rotary) for i in crossatt_layers})
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

    def init_state(self, max_seqlen=1000, state=None, **kwargs):
        cache = DynamicCache()
        return cache

    def step(self, y, ctx, time_step, cache):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y, cache=cache, time_step=time_step, layer_idx=i)
            if i in self.crossatt_layers:
                v, att = self.cross_att[str(i)](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1), cache

