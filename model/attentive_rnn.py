from abc import abstractmethod

import torch


class AttentiveRNN(torch.nn.Module):
    @abstractmethod
    def forward(self, x, ctx, x_mask, ctx_mask):
        pass

    @abstractmethod
    def init_state(self):
        pass

    @abstractmethod
    def step(self, x, ctx, x_mask=None, y_mask=None):
        pass
