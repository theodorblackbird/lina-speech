import torch
from jsonargparse import ArgumentParser
from model.lina import Lina

def regotron_loss(att, delta=0.01):
    """
        See https://arxiv.org/abs/2204.13437 .
    """
    a_j = att * torch.arange(att.shape[1], device=att.device).unsqueeze(0)
    a_j = a_j.mean(1)
    l = a_j[:-1] - a_j[1:] + delta/a_j.shape[0]
    l = torch.max(l, torch.zeros_like(l)).sum()
    return float(l)
    
def sampling_heuristic(cuts, delta=0.01, frac=0.5):
    """
        Rejects frac% highest regotron losses (typically skip/repetition),
        then rejects frac% shortest generations (typically early stopping).
    """
    l = len(cuts)
    cuts.sort(key=lambda x: regotron_loss(x[1], delta=delta), reverse=True)
    cuts = cuts[int(l*frac):]
    l = len(cuts)
    cuts.sort(key=lambda x: x[0].shape[2])
    cuts = cuts[int(l*frac):]
    return cuts

def instantiate_load(config, state_dict):
    parser = ArgumentParser()
    parser.add_argument('--model', type=Lina)
    conf = parser.parse_path(config)
    model = parser.instantiate_classes(conf)["model"]
    state_dict = torch.load(state_dict)["model"]
    model.load_state_dict(state_dict)
    return model
