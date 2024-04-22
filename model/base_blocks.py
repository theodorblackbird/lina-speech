import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, rotary=True):
        super().__init__()
        self.qkv = nn.Linear(dim, 3*dim)
        assert dim % heads == 0
        self.heads = heads
        self.rotary = RotaryEmbedding((dim // heads) // 2) if rotary else None
    def forward(self, x, mask=None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        if self.rotary is not None:
            q, k = map(self.rotary.rotate_queries_or_keys, (q, k))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = rearrange(y, "b h n d -> b n (h d)")
        return y

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.p_in = nn.Linear(d_model, (d_model * 4 // 3) * 2)
        self.p_out = nn.Linear(d_model * 4 // 3, d_model)

    def forward(self, x):
        gate, x = self.p_in(x).chunk(2, dim=-1)
        return self.p_out(nn.functional.silu(gate) * x)


def unpack_ignore(x):
    return x[0] if type(x) is tuple else x

class MixingBlock(torch.nn.Module):
    def __init__(self, tmix: Callable, cmix: Callable, norm: Callable, dropout=0.):
        super().__init__()
        self.tmix = tmix()
        self.cmix = cmix()
        self.norm1 = norm()
        self.norm2 = norm()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        x = unpack_ignore(self.tmix(self.norm1(x), **kwargs)) + x
        x = self.cmix(self.norm2(x)) + x
        x = self.drop(x)
        return x
