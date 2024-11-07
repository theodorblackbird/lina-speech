import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from model.modeling_lina import LinaModel
from model.tools import delay_rvq, sequence_mask
from torch.nn.utils.rnn import pad_sequence
from functools import reduce

def filter_unk(x, tokenizer: PreTrainedTokenizerBase):
    try: 
        tokenizer.encode(x)
        return True
    except:
        return False

def speaker_state_dict(params: list[tuple[torch.Tensor]]) -> dict[str, torch.Tensor]:
    state_dict = {}
    for i, layer in enumerate(params):
        if len(layer) == 2:
            k, v = layer

            state_dict[f"layer{i}_k"] = k
            state_dict[f"layer{i}_v"] = v
        else:
            state_dict[f"layer{i}"] = layer
    return state_dict

def filter_except(x):
    try:
        tokenizer.encode(x)
        return True
    except:
        return False

def parse_speaker_state(path, device="cpu") -> list[torch.Tensor]:
    with safe_open(path, framework="pt", device=device) as state:
        keys = [k for k in state.keys() if k.endswith("_k")]
        keys.sort(key=lambda x: int("".join([xx for xx in x if xx.isdigit()])))
        params_list = []
        for k in keys:
            v = state.get_tensor(k[:-2] + "_v")
            k = state.get_tensor(k)
            params_list.append((k, v))
    return params_list


def simple_collate(batch: list[dict], tokenizer: PreTrainedTokenizerBase):
    audio_token, text = zip(*[(x["audio_token"], x["text"]) for x in batch])
    orig_token = audio_token

    audio_token_delayed = []
    for x in audio_token:
        x = x.squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = delay_rvq(x + 3, head_token=1, tail_token=2).transpose(-1,-2)
        audio_token_delayed.append(x)

    text_token = [torch.LongTensor(tokenizer.encode("[BOS]" + x + "[EOS]")) for x in text]
    xlen, ylen = map(lambda x: [xx.shape[0] for xx in x], (text_token, audio_token_delayed))
    x_mask, y_mask = map(lambda x: sequence_mask(x, device="cpu"), (torch.tensor(xlen), torch.tensor(ylen)))

    audio_token, text_token = map(lambda x: pad_sequence(x, batch_first=True, padding_value=0), (audio_token_delayed, text_token))
    encoder_mask = (x_mask.unsqueeze(1) * x_mask.unsqueeze(2))
    crossatt_mask = (x_mask.unsqueeze(1) * y_mask.unsqueeze(2))
    crossatt_mask[:, :, 0] = True

    return {
        "text_token": text_token,
        "audio_token": audio_token,
        "orig_token": orig_token,
        "crossatt_mask": crossatt_mask,
        "encoder_mask": encoder_mask,
        "text": text,
        "y_mask": y_mask,
        "x_len": xlen,
        "y_len": ylen,
    }
bandwidth_id = torch.tensor(0)

def train_initial_state(
    model: LinaModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int,
    lr: float=0.1,
    grad_acc:int=4,
    batch_size:int=2,
    scale:float=0.02,
    save_every_k_steps:int=0,
    seed:int=123,
    rank:int=1,
):
    if save_every_k_steps > 0:
        save_every_k_steps_params = []

    model.attentive_rnn.to_mode("fused_recurrent")
    model = model.train()
    parameters = model.attentive_rnn.get_init_state_tuning_params(lora=rank, device="cuda")
    optimizer = torch.optim.Adam(reduce(tuple.__add__, parameters), lr=lr)

    def inf_sampler_wo_replacement(length):
        random.seed(seed)
        while True:
            idx = list(range(length))
            random.shuffle(idx)
            for i in idx:
                yield i

    def model_step(model, batch, parameters, batch_size):
        batch = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items()}
        text_token = batch["text_token"]
        audio_token = batch["audio_token"][..., :]
        crossatt_mask = batch["crossatt_mask"]
        encoder_mask = batch["encoder_mask"]
        y_mask = batch["y_mask"]
        x_len, y_len = batch["x_len"], batch["y_len"]

        init_state = model.attentive_rnn.get_state_from_params(parameters, batch_size, scale=scale)


        logits, loss, att, _, _ = model(text_token, audio_token, encoder_mask, crossatt_mask, logits_mask=y_mask, init_state=init_state)

        return loss

    train_dl = iter(DataLoader(dataset, 
                               batch_size=batch_size, 
                               sampler=inf_sampler_wo_replacement(len(dataset)),
                               num_workers=1,
                               collate_fn=lambda x: simple_collate(x, tokenizer)))

    train_losses = []

    k_steps = 0
    for i in tqdm(range(n_samples//batch_size)):

        batch = next(train_dl)

        loss = model_step(model, batch, parameters, batch_size)
        train_losses.append(loss.item())
        loss.backward()

        if i % grad_acc == grad_acc - 1:
            optimizer.step()
            optimizer.zero_grad()
            k_steps += 1
            if save_every_k_steps>0:
                if k_steps%save_every_k_steps==0:
                    save_every_k_steps_params.append(deepcopy(parameters))


    if save_every_k_steps > 0:
        save_every_k_steps_params.append(parameters)
        parameters = save_every_k_steps_params
    model = model.eval()
    return parameters, train_losses



