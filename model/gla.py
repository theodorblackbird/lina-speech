import torch
from einops import rearrange
from torch import nn

from model.attentive_rnn import AttentiveRNN
from model.crossatt import CrossAttention, BlindCrossAttention
from model.gptblock import GPTBlock
from fla.layers import GatedLinearAttention


def exists(x):
    return x is not None


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.p_in = nn.Linear(d_model, (d_model * 4 // 3) * 2)
        self.p_out = nn.Linear(d_model * 4 // 3, d_model)

    def forward(self, x):
        gate, x = self.p_in(x).chunk(2, dim=-1)
        return self.p_out(nn.functional.silu(gate) * x)


class AttentiveGLA(AttentiveRNN):
    def __init__(
        self,
        d_model,
        n_layer,
        d_context,
        heads,
        dropout_att=0.0,
        d_blind=64,
        blind=False,
    ):
        super().__init__()
        block = lambda d, h: GPTBlock(lambda: GatedLinearAttention(d_model=d, num_head=h),
                                                   lambda: SwiGLU(d),
                                                   lambda: nn.LayerNorm(d),
                                                   )
        self.encoder = nn.Sequential(*[block(d_model, heads) for i in range(n_layer)])
        self.decoder = nn.Sequential(*[block(d_model, heads) for i in range(n_layer)])
        self.cross_att = (
            BlindCrossAttention(
                d_model, d_context, d_model, 1, block(d_blind, 1), dropout_att
            )
            if blind
            else CrossAttention(d_model, d_context, d_model, heads, dropout_att)
        )

    def forward(self, x, ctx, x_mask=None, ctx_mask=None):
        if exists(x_mask) and exists(ctx_mask):
            mask = rearrange(x_mask, "b i -> b i 1") * rearrange(
                ctx_mask, "b j -> b 1 j"
            )
            mask = rearrange(mask, "b i j -> b 1 i j")
        else:
            mask = None

        y = self.encoder(x)
        v, att = self.cross_att(y, ctx, mask=mask)
        y = self.decoder(y + v)

        return y, att

    def init_state(self, max_seqlen=1000):
        pass

    def step(self, y_embd, x_enc, time_step):
        pass


