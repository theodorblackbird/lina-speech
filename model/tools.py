from typing import Callable, List, Optional
from itertools import accumulate

import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_2d_sequence(seq, padding_value=0):
    max_x, max_y = map(max, zip(*map(lambda x: x.shape, seq)))
    pad = lambda x: torch.nn.functional.pad(
        x,
        (0, max_y - x.shape[1], 0, max_x - x.shape[0]),
        value=padding_value,
    )
    return torch.stack([pad(x) for x in seq])

def packmask_2d(xlen: list[int], ylen: list[int], offset: int=0) -> torch.Tensor:
    _, ybound = map(lambda x: [0] + list(accumulate(x, int.__add__)), (xlen, ylen))
    lb, hb = [], []

    for n, l, h in zip(xlen, ybound[:-1], ybound[1:]):
        lb += [l]*n
        hb += [h]*n

    lb, hb = map(torch.tensor, (lb, hb))
    if offset:
        lb -= offset
        hb += offset

    rge = torch.arange(ybound[-1])

    lm = rge.unsqueeze(0) >= lb.unsqueeze(1)
    hm = rge.unsqueeze(0) < hb.unsqueeze(1)

    return lm * hm


def topk_sampling(seq, k=1, temp=1.):
    topk = torch.topk(seq, k, dim=-1)
    logits = seq / temp
    mask = logits < topk.values[:, [-1]]
    logits[mask]  = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def delay_rvq(
    code,
    head_token: int = -2,
    tail_token: int = -3,
):
    q, _ = code.shape
    extension = torch.ones((q, q + 1)).tril() * head_token
    extension += torch.ones((q + 1, q)).tril(diagonal=-1).T * tail_token
    extension = torch.flip(extension, (1,))
    extended_code = torch.cat((code, extension), axis=1)
    for i in range(q):
        extended_code[i, :] = torch.roll(extended_code[i, :], i + 1)

    return extended_code.long()

def undelay_rvq(extended_code):
    q, _, n = extended_code.shape
    out = []
    for i in range(q):
        out.append(torch.roll(extended_code[i], -(i + 1), dims=1))
    out = torch.stack(out, dim=0)
    return out[:, :, :-(q+1)]

def sequence_mask(lengths, max_len=None, device=default_device):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask
