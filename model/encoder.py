import math
import random
from typing import Optional

from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, einsum
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
            rotary=True,
            ):
        super().__init__()
        self.sa = torch.nn.ModuleList(
                [
                    MixingBlock(lambda: SelfAttention(dim, heads, rotary=rotary),
                                lambda: SwiGLU(dim),
                                lambda: torch.nn.LayerNorm(dim),
                                dropout)
                    for _ in range(n_layers)
                    ]
                )


    def forward(self, x, mask=None, pos=None):
        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m")
            mask = torch.logical_or(mask, rearrange(torch.eye(mask.shape[-1], device=x.device), "n m ->1 1 n m"))


        for block in self.sa:
            x = block(x, mask=mask, pos=pos)
        return x

class SimpleSpeakerEncoder(torch.nn.Module):
    def __init__(
            self,
            dim:int,
            dim_inner:int,
            heads:int,
            n_layers=6,
            dropout=0.1,
            rotary=True,
            window_length: int = 256,
            rank: int = 1,
            ):
        super().__init__()
        self.sa = torch.nn.ModuleList(
                [
                    MixingBlock(lambda: SelfAttention(dim_inner, heads, rotary=rotary),
                                lambda: SwiGLU(dim_inner),
                                lambda: torch.nn.LayerNorm(dim_inner),
                                dropout)
                    for _ in range(n_layers)
                    ]
                )
        self.window_length = window_length
        self.in_proj = nn.Linear(dim, dim_inner)
        self.out_proj = nn.Linear(dim_inner, dim)

    def forward(self, x, avoid_n_first_frames: int = 150, **kwargs):
        winl = self.window_length
        if self.training:
            b, n, d = x.shape
            i = random.randint(avoid_n_first_frames, n - winl)
            x = x[:, i:i+winl]
        else:
            x = x[:, 0:winl]
        x = self.in_proj(x)
        for block in self.sa:
            x = block(x)
        x = x[:, 0]
        x = self.out_proj(x)
        return x
