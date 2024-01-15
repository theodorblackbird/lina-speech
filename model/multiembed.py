from functools import partial

import torch
from torch import nn


class MultiEmbedding(nn.Module):
    """
    Stacks multiple homogeneous embedding (same size, same embedding dim.) into one single weight.
    """

    def __init__(self, n_level, n_emb, d_emb, padding_idx=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n_level, n_emb, d_emb, requires_grad=True)
        )
        self.n_level = n_level
        self.padding_idx = padding_idx
        nn.init.normal_(self.weight)

    def forward(self, idx):
        emb_fn = partial(nn.functional.embedding, padding_idx=self.padding_idx)
        return torch.vmap(emb_fn)(idx, self.weight)


if __name__ == "__main__":
    b, n, q, l, d = 2, 3, 2, 2, 2
    e = torch.compile(MultiEmbedding(q, l, d))
    i = torch.randint(l, (q, b, n))
    print(i)
    print(e(i).shape)
    print(e(i))
