from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttentionPP, CrossAttention, BlindCrossAttention
from model.base_blocks import MixingBlock, SwiGLU

from typing import Optional, Tuple, List
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.utils.generation import InferenceParams
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache



class AttentiveMamba(AttentiveRNN):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            heads: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dropout_att: float = 0.0,
            dropout: float = 0.,
            d_blind: int = None,
            blind: bool = False,
            cross_att_pp: bool=False,
            rotary: bool = False,
            pos_type="sinusoidal",
            version: int = 2,
            ):
        super().__init__()
        if version == 1:
            mamba = Mamba
        elif version == 2:
            mamba = Mamba2
        else:
            raise NotImplementedError

        block = lambda d, h, i: MixingBlock(lambda: mamba(d_model=d_model,
                                                       d_state=d_state,
                                                       d_conv=d_conv,
                                                       expand=expand,
                                                       layer_idx=i,
                                                       ),
                                         lambda: SwiGLU(d),
                                         lambda: nn.LayerNorm(d),
                                         dropout=dropout,
                                         )
        self.encoder = nn.ModuleList([block(d_model, heads, i) for i in range(n_layer)])
        self.decoder = nn.ModuleList([block(d_model, heads, i + n_layer) for i in range(n_layer)])
        if d_blind is None:
            d_blind=d_model

        if blind:
            self.cross_att = BlindCrossAttention(d_model, d_model, d_model, 1, block(d_blind, heads, 2*n_layer), dropout_att, pos_dim=d_blind, rotary=rotary, pos_type=pos_type)
        elif cross_att_pp:
            self.cross_att = CrossAttentionPP(d_model, block(d_model, heads, 2*n_layer), 1,ca_dropout=dropout_att)
        else:
            self.cross_att = CrossAttention(d_model, d_model, d_model, heads, dropout_att)



    def forward(self, x, ctx, mask=None, pos=None, attention_only=False, forced_attention=None):
        for e in self.encoder:
            #x = torch.utils.checkpoint.checkpoint(e, x, use_reentrant=False)
            x = e(x)

        if forced_attention is not None:
            ctx = self.cross_att.ln_v(self.cross_att.v(ctx))
            v = torch.matmul(forced_attention.transpose(1, 2), ctx)
            att = forced_attention
        else:
            v, att = self.cross_att(x, ctx, mask=mask)
        if attention_only:
            return v, att
        x = x + v
        for d in self.decoder:
            #x = torch.utils.checkpoint.checkpoint(d, x, use_reentrant=False)
            x = d(x)
        return x, att

    def init_state(self, max_seqlen=1000, state=None):
        if state is None:
            self.inference_params = InferenceParams(
                max_seqlen=max_seqlen, max_batch_size=1, seqlen_offset=1
            )
        else:
            self.inference_params = state
        n_layer = len(self.encoder)
        for i in range(n_layer):
            self.encoder[i].tmix.layer_idx = i
            self.decoder[i].tmix.layer_idx = i + n_layer
            self.cross_att.pos_net.tmix.layer_idx = 2 * n_layer

    def get_state(self):
        return self.inference_params


    def step(self, y_embd, x_enc, time_step):
        for e in self.encoder:
            y_embd = e(y_embd, inference_params=self.inference_params)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step, inference_params=self.inference_params)
        y_embd = y_embd + v
        for d in self.decoder:
            y_embd = d(y_embd, inference_params=self.inference_params)
        return y_embd, att

class CrossAttMambaV2(AttentiveRNN):
    def __init__(
            self,
            n_layer,
            crossatt_layers: List[int],
            d_model,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            crossatt_heads: int = 2,
            dropout_att: float = 0.1,
            rotary: bool = False,
            version: int = 2,
            ):
        super().__init__()

        if version == 1:
            mamba = Mamba
        elif version == 2:
            mamba = Mamba2
        else:
            raise NotImplementedError

        block = lambda: MixingBlock(lambda: mamba(d_model=d_model,
                                                       d_state=d_state,
                                                       d_conv=d_conv,
                                                       expand=expand,
                                                       ),
                                         lambda: SwiGLU(d_model),
                                         lambda: nn.LayerNorm(d_model),
                                         )
        self.blocks = nn.ModuleList([block() for i in range(n_layer)])
        self.cross_att = nn.ModuleDict({str(i): CrossAttention(d_model, d_model, d_model, crossatt_heads, dropout=dropout_att, rotary=rotary) for i in crossatt_layers})
        self.crossatt_layers = crossatt_layers
        self.n_layer = n_layer

    def forward(self, x, ctx, mask=None, pos=None, attention_only=False):
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
            if i >= max(self.crossatt_layers):
                return y, att
        return y, att

    def init_state(self, max_seqlen=1000):
        self.inference_params = InferenceParams(
            max_seqlen=max_seqlen, max_batch_size=1, seqlen_offset=1
        )
        n_layer = len(self.blocks)
        for i in range(n_layer):
            self.blocks[i].tmix.layer_idx = i


    def step(self, y, ctx, time_step):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y, inference_params=self.inference_params)
            if i in self.crossatt_layers:
                v, att = self.cross_att[str(i)](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1)


class CrossAttMamba(AttentiveRNN):
    def __init__(
            self,
            n_layer,
            d_model,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            crossatt_heads: int = 2,
            n_crossatt_layer: int = 2,
            dropout_att: float = 0.1,
            rotary: bool = False,
            version: int = 2,
            ):
        super().__init__()

        if version == 1:
            mamba = Mamba
        elif version == 2:
            mamba = Mamba2
        else:
            raise NotImplementedError

        block = lambda: MixingBlock(lambda: mamba(d_model=d_model,
                                                       d_state=d_state,
                                                       d_conv=d_conv,
                                                       expand=expand,
                                                       ),
                                         lambda: SwiGLU(d_model),
                                         lambda: nn.LayerNorm(d_model),
                                         )
        self.blocks = nn.ModuleList([block() for i in range(n_layer)])
        self.cross_att = nn.ModuleList([CrossAttention(d_model, d_model, d_model, crossatt_heads, dropout=dropout_att, rotary=rotary) for i in range(n_crossatt_layer)])
        self.cross_att_layers = list(range(n_layer//2-(n_crossatt_layer//2), n_layer//2 + (n_crossatt_layer//2) + int(n_crossatt_layer%2)))
        print(self.cross_att_layers)
        self.n_layer = n_layer

    def forward(self, x, ctx, mask=None, pos=None):
        y = x
        att = None
        for i, b in enumerate(self.blocks):
            y = torch.utils.checkpoint.checkpoint(b, y, use_reentrant=False)
            if i in self.cross_att_layers:
                v, _att = self.cross_att[i - self.cross_att_layers[0]](y, ctx, mask=mask.unsqueeze(1))
                y = y + v
                if _att is not None:
                    if att is not None:
                        att = torch.cat((att, _att), dim=1)
                    else:
                        att = _att
        return y, att

    def init_state(self, max_seqlen=1000):
        self.inference_params = InferenceParams(
            max_seqlen=max_seqlen, max_batch_size=1, seqlen_offset=1
        )
        n_layer = len(self.blocks)
        for i in range(n_layer):
            self.blocks[i].tmix.layer_idx = i


    def step(self, y, ctx, time_step):
        atts = []
        for i, b in enumerate(self.blocks):
            y = b(y, inference_params=self.inference_params)
            if i in self.cross_att_layers:
                v, att = self.cross_att[i - self.cross_att_layers[0]](y, ctx, time_step=time_step)
                atts.append(att)
                y = y + v
        return y, torch.cat(atts, dim=1)
