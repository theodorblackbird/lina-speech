from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from model.attentive_rnn import AttentiveRNN
from model.multiembed import MultiEmbedding
from torch import Tensor, nn
from .tools import sequence_mask, topk_sampling, undelay_rvq

def exists(x):
    return x is not None
class LinaModel(nn.Module):
    def __init__(
        self,
        attentive_rnn: AttentiveRNN,
        d_model: int,
        n_quant: int,
        n_codebook: int,
        n_special_token_in: int,
        n_special_token_out: int,
        n_txt_vocab: int,
        tie_embed: bool = False,
        txt_encoder: Optional[nn.Module] = None,
        spk_encoder: Optional[nn.Module] = None,
        mask_text_p: float = 0.,
    ):
        super(LinaModel, self).__init__()

        self.n_quant = n_quant
        self.n_codebook = n_codebook
        self.n_special_token_in = n_special_token_in
        self.n_special_token_out = n_special_token_out
        self.mask_text_p = mask_text_p
        self.n_txt_vocab = n_txt_vocab + int(mask_text_p > 0.)
        self.n_target_vocab = n_codebook + n_special_token_out  # no padding token

        self.txt_encoder = txt_encoder
        self.spk_encoder = spk_encoder
        self.attentive_rnn = attentive_rnn

        self.txt_embed = nn.Embedding(
            n_txt_vocab,
            d_model,
            padding_idx=0,
        )
        self.rvq_embed = MultiEmbedding(self.n_quant, n_codebook + n_special_token_in, d_model, padding_idx=0) 


        self.logits_head = EinMix(
            "b n d -> b n q l",
            weight_shape="q l d",
            q=self.n_quant,
            d=d_model,
            l=self.n_target_vocab,
        )
        if tie_embed:
            self.logits_head.weight = self.rvq_embed.weight

    def forward(self, x, y, encoder_mask, crossatt_mask, logits_mask=None, attention_only=False, forced_attention=None, init_state=None, crossatt_pos=None):
        # b: batch
        # n: seq length
        # q: quantizers
        # d: model dim
        # l: target vocab
        
        if self.mask_text_p > 0.:
            mask = torch.empty(x.shape[0]).bernoulli_(self.mask_text_p)
            x[mask] = self.n_txt_vocab - 1
        
        x_embd = self.txt_embed(x)
        y_embd = self.rvq_embed(rearrange(y, "b n q -> q b n"))
        q, b, n, d = y_embd.shape
        y_embd = reduce(y_embd, "q b n d -> b n d", "sum", q=q)
        
        x_enc = self.txt_encoder(x_embd, mask=encoder_mask)

        if self.spk_encoder is not None:
            spk_embd = self.spk_encoder(y_embd)
            y_embd[:,0] = spk_embd

        y_hat, att = self.attentive_rnn(
            y_embd[:, :-1, :],
            x_enc,
            mask=crossatt_mask[:,:-1],
            forced_attention=forced_attention[:,:,:y_embd.shape[1]-1] if forced_attention is not None else None,
            attention_only=attention_only,
            init_state=init_state,
            crossatt_pos=crossatt_pos,
        )
        if attention_only:
            return att
        logits = self.logits_head(y_hat)
        if logits_mask is not None:
            masked_logits = logits[logits_mask[:, 1:], :, :]
            masked_target = y[:, 1:][logits_mask[:, 1:], :]
            flat_logits = rearrange(masked_logits, "n q l -> (n q) l")
            flat_target = rearrange(masked_target, "n q   -> (n q)")
        else:
            masked_logits = logits 
            masked_target = y[:, 1:]
            flat_logits = rearrange(masked_logits, "b n q l -> (b n q) l")
            flat_target = rearrange(masked_target, "b n q   -> (b n q)")

        loss = F.cross_entropy(flat_logits, flat_target, ignore_index=1)

        return logits, loss, att, masked_logits, masked_target


    @torch.inference_mode()
    def generate_batch(
        self,
        x: Tensor,
        batch_size: int=3,
        prompt: Optional[Tensor] = None,
        device: str = "cpu",
        max_seqlen: int = 1000,
        k: int = 100,
        first_greedy_quant: int = 1,
        temp: float = 1.0,
        init_state: Optional[dict] = None,
        force_max_seqlen: bool = False,
    ):
        x = repeat(x, "n -> b n", b=batch_size).to(device)
        stop_token = torch.ones(self.n_quant, 1, 1, device=device) * 2
        all_stop_token = torch.zeros(batch_size, 1, device=device).bool()
        y_start = torch.ones(self.n_quant, batch_size, 1, device=device).long()

        x_embd = self.txt_embed(x)
        y_embd = reduce(self.rvq_embed(y_start), "q b n d -> b n d", "sum")
        
        p_len = -1
        if exists(prompt):
            if prompt.shape[1] != batch_size:
                prompt = repeat(prompt, "q 1 n -> q b n", b=batch_size) + 3
            prompt = reduce(self.rvq_embed(prompt.to(device)), "q b n d -> b n d", "sum")
            p_len = prompt.shape[1]

            if self.spk_encoder is not None:
                spk_embd = self.spk_encoder(prompt)
                prompt[:,0] = spk_embd


        x_enc = self.txt_encoder(x_embd)
        state = init_state
        if state is None:
            state = self.attentive_rnn.init_state(max_seqlen=max_seqlen, batch_size=batch_size)

        qs, atts, stop_tokens = [], [], []

        for t in range(max_seqlen):
            y_embd, att, state= self.attentive_rnn.step(y_embd, x_enc, t, state)
            atts.append(att)
            logits = self.logits_head(y_embd)
            logits = rearrange(logits, "b 1 q l -> q b l")
            q_sampled = []

            for i, q in enumerate(logits):
                q_sampled.append(
                    topk_sampling(q, k=k, temp=temp)
                    if i < first_greedy_quant
                    else topk_sampling(q, k=1)
                )
            q_sampled = torch.stack(q_sampled)
            qs.append(q_sampled)

            is_stop_token = (q_sampled == stop_token).prod(dim=0)
            stop_tokens.append(is_stop_token)
            all_stop_token.logical_or_(is_stop_token)

            if all_stop_token.prod() and not force_max_seqlen:
                break

            if exists(prompt) and t < p_len:
                y_embd = prompt[:, [t]]
            else:
                y_embd = self.rvq_embed(q_sampled)
                y_embd = reduce(y_embd, "q b n d -> b n d", "sum")

        atts =  torch.cat(atts, dim=2) if exists(atts[0]) else None
        qs = torch.stack(qs, dim=2).squeeze(-1)
        stop_tokens.append(torch.ones(batch_size, 1, device=device))
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze(-1)
        b, n = stop_tokens.shape
        rvq = (undelay_rvq(qs) - self.n_special_token_in).clamp_min(0)
        stop_idx = (stop_tokens * rearrange(torch.arange(n, device=stop_tokens.device), "n -> 1 n")).long()
        cuts = []
        for i in range(stop_idx.shape[0]):
            idx = torch.unique(stop_idx[i])[1]
            cuts.append((rvq[:,[i], :idx-self.n_quant], atts[i, :, :idx]))
        return qs, atts, stop_tokens, cuts


