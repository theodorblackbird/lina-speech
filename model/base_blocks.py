import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, rotary=True, is_causal=False):
        super().__init__()
        self.qkv = nn.Linear(dim, 3*dim)
        assert dim % heads == 0
        self.heads = heads
        self.rotary = RotaryEmbedding((dim // heads) // 2) if rotary else None
        self.is_causal = is_causal

    def forward(self, x, mask=None, pos=None, cache=None, layer_idx=None, time_step=0):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        
        if cache is not None:
            assert layer_idx is not None
            cache.update(k, v, layer_idx=layer_idx)
            k, v = cache[layer_idx]

        if self.rotary is not None:
            if pos is not None:
                q, k = map(lambda x: apply_rotary_emb(self.rotary(pos).unsqueeze(1), x), (q, k))
            else:
                #q, k = map(lambda x: self.rotary.rotate_queries_or_keys(x, offset=time_step), (q, k))
                q = self.rotary.rotate_queries_or_keys(q, offset=time_step)
                k = self.rotary.rotate_queries_or_keys(k)


        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.is_causal)
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
    def __init__(self, tmix: Callable, cmix: Callable, norm: Callable, dropout: float=0.):
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
