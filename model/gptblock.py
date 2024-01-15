import torch
from typing import Callable

class GPTBlock(torch.nn.Module):
    def __init__(self, att: Callable, ffn: Callable, norm: Callable):
        super().__init__()
        self.att = att()
        self.ffn = ffn()
        self.norm1 = norm()
        self.norm2 = norm()

    def forward(self, x):
        x = self.att(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x
