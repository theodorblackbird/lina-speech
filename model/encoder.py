import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import EinMix
from rotary_embedding_torch import RotaryEmbedding


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, ffn_factor=3, dropout=0.1, rotary=True):
        super().__init__()
        self.qkv = EinMix(
            "b n d -> qkv b n dd",
            weight_shape="qkv d dd",
            qkv=3,
            d=dim,
            dd=dim,
        )
        assert dim % heads == 0
        self.heads = heads
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_factor, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.rotary = RotaryEmbedding((dim // heads) // 2) if rotary else None

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            mask = mask.unsqueeze(1)
            mask = mask.masked_fill(~mask, -torch.finfo(x.dtype).max)
        q, k, v = self.qkv(self.ln1(x))
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        if self.rotary is not None:
            q, k = map(self.rotary.rotate_queries_or_keys, (q, k))
        xx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = rearrange(xx, "b h n d -> b n (h d)") + x
        x = x + self.ffn(self.ln2(x))
        return x


def exists(x):
    return x is not None


class ConvNeXt(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        depthwise=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim if depthwise else 1,
            padding="same",
        )
        self.p1 = nn.Linear(dim, 3 * dim)
        self.p2 = nn.Linear(3 * dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        res = x
        if exists(mask):
            x = x * mask.unsqueeze(-1)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.ln(x)
        x = self.p2(F.gelu(self.p1(x)))
        x = x + res

        return x


class ConvNet(nn.Module):
    def __init__(self, dim, n_layers, **kwargs):
        super().__init__()
        self.conv = nn.ModuleList([ConvNeXt(dim, **kwargs) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for c in self.conv:
            x = x + c(x, mask=mask)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        prenet: Optional[nn.Module] = None,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.prenet = prenet
        self.sa = nn.ModuleList(
            [SelfAttentionBlock(dim, heads) for _ in range(n_layers)]
        )

    def forward(self, x, mask=None):
        if self.prenet is not None:
            x = self.prenet(x, mask=mask)
        for block in self.sa:
            x = block(x, mask=mask)
        return x
