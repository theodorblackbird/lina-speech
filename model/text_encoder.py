import math

import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


def exists(x):
    return x is not None


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = self.c_fc(x)
        x = F.gelu(x)
        if mask is not None:
            x = x * mask
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias=True, pos_emb=None):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        if exists(pos_emb):
            self.pos_emb = pos_emb

    def forward(self, x, attn_mask=None):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if hasattr(self, "pos_emb"):
            q = self.pos_emb.rotate_queries_or_keys(q)
            k = self.pos_emb.rotate_queries_or_keys(k)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if exists(attn_mask):
            att = att.masked_fill(~attn_mask, -1e4)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class AttBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias=False, pos_emb=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(
            n_embd,
            n_head,
            dropout,
            bias=bias,
            pos_emb=pos_emb,
        )
        self.n_head = n_head
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout, bias=bias)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = rearrange(mask, "b n -> b n 1")
            x = x * mask
            attn_mask = (mask * mask.transpose(1, 2)).unsqueeze(1)
        else:
            attn_mask = None

        y = self.attn(x, attn_mask=attn_mask)
        x = self.ln_1(x + y)
        y = self.ff(x, mask=mask)
        x = self.ln_2(x + y)

        if mask is not None:
            x = x * mask
        return x


class TextEncoder(nn.Module):
    def __init__(self, n_embd, n_head, dropout, n_blocks=4, max_len=200) -> None:
        super().__init__()
        self.att_blocks = nn.ModuleList(
            [AttBlock(n_embd, n_head, dropout) for _ in range(n_blocks)]
        )
        self.position_emb = nn.Embedding(max_len, n_embd)
        self.convs = nn.ModuleList(
            [nn.Conv1d(n_embd, n_embd, 5, padding="same") for _ in range(3)]
        )
        self.convs_norm = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(3)])

    def forward(self, x, mask=None):
        if mask is not None:
            x *= mask.unsqueeze(2)
        for c, n in zip(self.convs, self.convs_norm):
            y = c(x.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                y *= mask.unsqueeze(2)
            y = n(y)
            y = nn.functional.relu(y)
            x = y + x
        for b in self.att_blocks:
            x = b(x, mask=mask)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


class TextEncoderV2(nn.Module):
    def __init__(self, dim, n_head, dropout, n_blocks=4) -> None:
        super().__init__()

        self.pos_emb = RotaryEmbedding(dim=(dim // n_head) // 2)
        self.att_blocks = nn.ModuleList(
            [
                AttBlock(dim, n_head, dropout, pos_emb=self.pos_emb)
                for _ in range(n_blocks)
            ]
        )
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, 5, padding="same") for _ in range(3)]
        )
        self.convs_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])

    def forward(self, x, mask=None):
        if mask is not None:
            x *= mask.unsqueeze(2)
        for c, n in zip(self.convs, self.convs_norm):
            y = c(x.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                y *= mask.unsqueeze(2)
            y = n(y)
            y = nn.functional.relu(y)
            x = y + x
        for b in self.att_blocks:
            x = b(x, mask=mask)
        return x
