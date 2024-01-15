from typing import Optional, List
from einops import repeat, rearrange
import torch
import torch.nn.functional as F


def exists(x):
    return x is not None


class MulticlassAccuracy:
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        ignore_index: Optional[List[int]] = None,
    ):
        self.num_classes = num_classes
        self.top_k = top_k
        self.ignore_index = ignore_index

    def __call__(self, preds, targets):
        n, c = preds.shape
        if exists(self.ignore_index):
            mask = ~torch.isin(targets, torch.tensor(self.ignore_index).to(targets))
            preds_top_k = torch.argsort(preds[mask, :], dim=-1, descending=True)
            n = preds_top_k.shape[0]
            targets = targets[mask]
        else:
            preds_top_k = torch.argsort(preds, dim=-1, descending=True)
        preds_top_k = (preds_top_k[:, : self.top_k] == targets.unsqueeze(1)).any(dim=1)
        return preds_top_k.sum() / len(preds_top_k)


if __name__ == "__main__":
    a = torch.randn(1000, 10)
    b = torch.cat((torch.zeros(999), torch.ones(1)))
    acc = MulticlassAccuracy(10, top_k=5, ignore_index=[0])
    print(acc(a, b))
