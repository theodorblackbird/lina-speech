import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import EinMix
from rotary_embedding_torch import RotaryEmbedding
from .base_blocks import MixingBlock, SwiGLU, SelfAttention



class TextEncoder(torch.nn.Module):
    def __init__(
            self,
            dim:int,
            heads:int,
            n_layers=4,
            dropout=0.1,
            ):
        super().__init__()
        self.sa = torch.nn.ModuleList(
                [
                    MixingBlock(lambda: SelfAttention(dim, heads),
                                lambda: SwiGLU(dim),
                                lambda: torch.nn.LayerNorm(dim),
                                dropout)
                    for _ in range(n_layers)
                    ]
                )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            mask = mask.unsqueeze(1)
            mask = mask.masked_fill(~mask, -torch.finfo(x.dtype).max)
        for block in self.sa:
            x = block(x, mask=mask)
        return x
