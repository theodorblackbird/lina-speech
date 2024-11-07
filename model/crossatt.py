import math

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


def exists(x):
    return x is not None


def scaled_dot_product_attention(query, key, value, mask=None):
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if exists(mask):
        attn_weight.masked_fill_(~mask, -torch.finfo(attn_weight.dtype).max)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value, attn_weight

class ConvPos(nn.Module):
    def __init__(self, dim, max_seq_len=2000, kernel_size=31):
        super().__init__()
        self.embed = nn.Embedding(max_seq_len, dim)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same")
        
    def forward(self, x):
        y = self.embed(x)
        y = rearrange(y, "b n c -> b c n")
        y = self.dw_conv(y)
        y = rearrange(y, "b c n -> b n c")
        return y


class SinPos(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        exp = torch.arange(self.dim // 2, device=x.device)
        exp = 2 * exp / (self.dim)
        exp = rearrange(exp, "e -> 1 1 e")
        x = rearrange(x, "b p -> b p 1")
        pos = x * torch.pow(10000, -exp)
        pos = torch.cat((pos, pos + math.pi / 2), dim=2)
        pos = torch.sin(pos)
        return pos

class CrossAttentionPP(nn.Module):
    def __init__(self, dim, inter_net, ca_heads, ca_dropout=0., max_seqlen=512):
        super().__init__()
        self.ca = nn.ModuleList([CrossAttention(dim, dim, dim, ca_heads, ca_dropout) for _ in range(2)])
        self.inter_net = inter_net
        self.pos_emb = nn.Embedding(max_seqlen, dim)

    def forward(self, q, k, mask=None, time_step=None, **kwargs):
        b, n, d = k.shape
        pos = torch.arange(n, device=k.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        if mask is not None:
            mask = mask.unsqueeze(1)
        k_pos = k + pos_emb
        y, att1 = self.ca[0](q, k, k_pos, mask=mask, time_step=time_step)
        y = self.inter_net(y, **kwargs)
        y, att2 = self.ca[1](y, k_pos, k, mask=mask, time_step=time_step)
        if att1 is not None:
            att = torch.cat((att1, att2), dim=1)
        else:
            att = None

        return y, att

        

class BlindCrossAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        att_dim,
        heads,
        pos_net,
        dropout=0.1,
        pos_dim=64,
        rotary=False,
        pos_type="sinusoidal",
    ):
        super().__init__()
        self.q = nn.Linear(q_dim, att_dim)
        self.k = nn.Linear(k_dim, att_dim)
        self.v = nn.Linear(k_dim, att_dim)
        self.pos_net = pos_net
        if pos_type=="sinusoidal":
            self.pos_embed = SinPos(pos_dim)
        elif pos_type=="convolutional":
            self.pos_embed = ConvPos(pos_dim)
        assert att_dim % heads == 0
        self.ln_q = nn.LayerNorm(att_dim)
        self.ln_k = nn.LayerNorm(att_dim)
        self.ln_v = nn.LayerNorm(att_dim)
        self.rotary = RotaryEmbedding((att_dim // heads) // 2) if rotary else None
        self.dropout_att = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        mask=None,
        time_step=None,
        pos=None,
        **kwargs,
    ):
        q = self.ln_q(self.q(q))
        v = self.ln_v(self.v(k))
        k = self.ln_k(self.k(k))

        q, k, v = map(lambda x: rearrange(x, "b n d -> b 1 n d"), (q, k, v))
        b, h, j, d = k.shape


        if mask is not None:
            mask = mask.unsqueeze(1)

        if pos is None:
            pos = torch.arange(j, device=k.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)

        if self.rotary is not None:
            if exists(time_step):
                q = self.rotary.rotate_queries_or_keys(q, offset=time_step)
                k = self.rotary.rotate_queries_or_keys(k)
            else:
                q, k = map(self.rotary.rotate_queries_or_keys, (q, k))

        if self.training:
            sdpa = lambda q, k, pos: (nn.functional.scaled_dot_product_attention(
                q, k, pos, attn_mask=mask, dropout_p=self.dropout_att.p
                ), None)
        else:
            sdpa = lambda q, k, pos: scaled_dot_product_attention(q, k, pos, mask=mask)
        
        x, att1 = sdpa(q, k, pos_emb.unsqueeze(1))
        x = rearrange(x, "b 1 n d -> b n d")
        x = self.pos_net(x, **kwargs)
        x = x[0] if type(x) is tuple else x
        x = rearrange(x, "b n d -> b 1 n d")
        pos_emb = rearrange(pos_emb, "b n d -> b 1 n d")
        x, att2 = sdpa(x, pos_emb, v)
        if att1 is not None:
            att = torch.cat((att1, att2), dim=1)
        else:
            att = None
        x = rearrange(x, "b 1 n d -> b n d")
        return x, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        k_dim,
        att_dim,
        heads,
        dropout=0.1,
        rotary=False,
    ):
        super().__init__()
        self.q = nn.Linear(q_dim, att_dim)
        self.k = nn.Linear(k_dim, att_dim)
        self.v = nn.Linear(k_dim, att_dim)
        assert att_dim % heads == 0
        self.heads = heads
        self.ln_q = nn.LayerNorm(att_dim)
        self.ln_k = nn.LayerNorm(att_dim)
        self.ln_v = nn.LayerNorm(att_dim)
        self.rotary = RotaryEmbedding((att_dim // heads) // 2) if rotary else None
        self.dropout_att = dropout

    def forward(
        self,
        q,
        k,
        v=None,
        mask=None,
        time_step=None,
        **kwargs,
    ):
        if v is None:
            v = k
        q = self.ln_q(self.q(q))
        v = self.ln_v(self.v(v))
        k = self.ln_k(self.k(k))
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        if self.rotary is not None:
            if exists(time_step):
                q = self.rotary.rotate_queries_or_keys(q, offset=time_step)
                k = self.rotary.rotate_queries_or_keys(k)
            else:
                q, k = map(self.rotary.rotate_queries_or_keys, (q, k))
        if self.training:
            x = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout_att
            )
            att = None
        else:
            x, att = scaled_dot_product_attention(q, k, v, mask=mask)
        x = rearrange(x, "b h n d -> b n (h d)")

        return x, att

if __name__ == "__main__":
    sinpos = SinPos(256)
    x = torch.arange(100)
    print(sinpos(x).shape)
    
