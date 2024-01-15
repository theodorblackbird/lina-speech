from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import EinMix
from model.attentive_rnn import AttentiveRNN
from model.multiembed import MultiEmbedding
from pytorch_lightning.cli import LightningCLI
from torch import Tensor, nn

torch.multiprocessing.set_sharing_strategy("file_system")
from .accuracy import MulticlassAccuracy
from .tools import sequence_mask, topk_sampling


def exists(x):
    return x is not None

class Lina(ptl.LightningModule):
    def __init__(
        self,
        attentive_rnn: AttentiveRNN,
        d_model: int,
        quant_layer: List[int],
        n_codebook: int,
        n_special_token_in: int,
        n_special_token_out: int,
        n_txt_vocab: int,
        tie_embed: bool = False,
        d_context: Optional[int] = None,
        txt_encoder: Optional[nn.Module] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
    ):
        super(Lina, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        self.n_quant = len(quant_layer)
        self.n_codebook = n_codebook
        self.n_special_token_in = n_special_token_in
        self.n_special_token_out = n_special_token_out
        self.n_txt_vocab = n_txt_vocab
        #TODO - not needed anymore
        self.n_target_vocab = n_codebook + n_special_token_out  # no padding token

        self.txt_encoder = txt_encoder if exists(txt_encoder) else nn.Identity()

        self.txt_embed = nn.Embedding(
            n_txt_vocab,
            d_context if exists(d_context) else d_model,
            padding_idx=0,
        )
        self.rvq_embed = MultiEmbedding(self.n_quant, n_codebook + n_special_token_in, d_model, padding_idx=0) 
        #nn.Embedding(
        #    n_codebook * self.n_quant + n_special_token_in,
        #    d_model,
        #)

        self.attentive_rnn = attentive_rnn

        self.logits_head = EinMix(
            "b n d -> b n q l",
            weight_shape="q l d",
            q=self.n_quant,
            d=d_model,
            l=self.n_target_vocab,
        )
        if tie_embed:
            self.logits_head.weight = self.rvq_embed.weight

        self.save_hyperparameters()  # ignore=["attentive_rnn", "txt_encoder"])

        self.accuracy_metric = MulticlassAccuracy(
            self.n_target_vocab,
            top_k=10,
            ignore_index=[0, 1],
        )

    def step(self, x, y, x_lens, y_lens):
        # b: batch
        # n: seq length
        # q: quantizers
        # d: model dim
        # l: target vocab
        x_embd = self.txt_embed(x)
        y_embd = self.rvq_embed(rearrange(y, "b n q -> q b n"))
        q, b, n, d = y_embd.shape
        y_embd = reduce(y_embd, "q b n d -> b n d", "sum", q=q)

        x_mask, y_mask = map(sequence_mask, (x_lens, y_lens))

        x_enc = self.txt_encoder(x_embd, x_mask)

        y_hat, att = self.attentive_rnn(
            y_embd[:, :-1, :],
            x_enc,
            y_mask[:, 1:],
            x_mask,
        )
        logits = self.logits_head(y_hat)

        masked_logits = logits[y_mask[:, 1:], :, :]
        # y = self.modulo_target(y)
        masked_target = y[:, 1:][y_mask[:, 1:], :]

        flat_logits = rearrange(masked_logits, "n q l -> (n q) l")
        flat_target = rearrange(masked_target, "n q   -> (n q)")

        loss = F.cross_entropy(flat_logits, flat_target)

        return logits, loss, att, masked_logits, masked_target

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_lens: Tensor,
        y_lens: Tensor,
    ):
        
        logits, loss, att, masked_logits, masked_target = self.step(x, y, x_lens, y_lens)

        accs = [
            self.accuracy_metric(
                masked_logits[:, i],
                masked_target[:, i],
            )
            for i in range(self.n_quant)
        ]

        return logits, loss, att, accs

    @torch.inference_mode()
    def generate(
        self,
        x: Tensor,
        teach_force: Optional[Tensor] = None,
        prompt: Optional[Tensor] = None,
        device: str = "cpu",
        max_seqlen: int = 1000,
        k: int = 100,
        first_greedy_quant: int = 1,
        temp: float = 1.0,
    ):
        x = rearrange(x, "n -> 1 n").to(device)
        y_start = torch.ones(1, 1, self.n_quant, device=device).long()

        x_embd = self.txt_embed(x).to(device)
        y_embd = self.rvq_embed(y_start)

        y_embd = reduce(y_embd, "b n q d -> b n d", "sum", q=self.n_quant)

        x_mask = torch.ones_like(x, device=device).bool()

        x_enc = self.txt_encoder(x_embd, x_mask)

        self.attentive_rnn.init_state(max_seqlen=max_seqlen)

        qs = []
        atts = []

        for t in range(max_seqlen):
            y_embd, att = self.attentive_rnn.step(y_embd, x_enc, time_step=t)
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
            qs.append(q_sampled.squeeze())

            if q_sampled.squeeze().prod() == 1:
                break

            q_sampled = rearrange(q_sampled, "q b 1 -> b 1 q")
            #TODO - multiembed
            y = 1 + q_sampled.where(
                q_sampled < 2,
                q_sampled
                + self.n_codebook * torch.arange(self.n_quant).to(device).long(),
            )
            if exists(teach_force):
                y_embd = reduce(
                    self.rvq_embed(teach_force[:, t]), "b q d -> b 1 d", "sum"
                )
            elif exists(prompt):
                if t < prompt.shape[1]:
                    y_embd = reduce(
                        self.rvq_embed(prompt[:, t]), "b q d -> b 1 d", "sum"
                    )
                else:
                    y_embd = reduce(self.rvq_embed(y), "b 1 q d -> b 1 d", "sum")
            else:
                y_embd = reduce(self.rvq_embed(y), "b 1 q d -> b 1 d", "sum")

        atts = torch.cat(atts, dim=2)
        return qs, atts

    def training_step(self, batch, _):
        x, y, x_len, y_len, code, phon, _ = batch
        logits, loss, att, accs = self(x, y, x_len, y_len)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for i, acc in enumerate(accs):
            self.log("train_acc_" + str(i), acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, idx):
        x, y, x_len, y_len, code, phon, _ = batch
        logits, loss, att, accs = self(x, y, x_len, y_len)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for i, acc in enumerate(accs):
            self.log("val_acc_" + str(i), acc, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        params = [
            {
                "params": self.parameters(),
                "weight_decay": self.weight_decay,
            }
        ]

        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=self.betas,
        )


def cli():
    class LinaCli(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.init_args.quant_layer", "model.init_args.quant_layer"
            )

    LinaCli(parser_kwargs={"parser_mode": "omegaconf"}, save_config_kwargs={"overwrite": True})
