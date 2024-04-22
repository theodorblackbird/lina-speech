from typing import Callable, List, Optional

import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def to_vocos(x):
    x = undelay_rvq(x.T)
    x = (x - 3).clamp_min(0)
    return x

def txt_to_phon(x):
    return espeak.phonemize([x], strip=True, njobs=1)[0]
    
def phon_to_code(x):
    y = [ds.vocab_x_code[t] for t in x]
    y = [ds.vocab_x_code["BOS"]] + y + [ds.vocab_x_code["EOS"]]
    return torch.tensor(y)

def sequence_mask(lengths, max_len=None, device=default_device):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def align_mask(src_mask, mel_mask):
    align_mask = torch.zeros(
        src_mask.shape[0], src_mask.shape[-1], mel_mask.shape[-1], dtype=torch.bool
    )
    for i, (src, mel) in enumerate(zip(src_mask, mel_mask)):
        w = torch.max(src.sum(-1))
        l = torch.max(mel.sum(-1))
        align_mask[i, :w, :l] = torch.ones(w, l, dtype=torch.bool)
    return align_mask


def last_that_fullfil(cond: Callable, x: torch.Tensor, strict: bool = True):
    res = cond(x).nonzero()
    if strict:
        assert len(res), f"no one fullfill {cond}"
    return res[-1]


def first_that_fullfil(cond: Callable, x: torch.Tensor, strict: bool = True):
    res = cond(x).nonzero()
    if strict:
        assert len(res), f"no one fullfill {cond}"
    return res[0]
