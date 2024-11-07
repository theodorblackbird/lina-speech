from typing import List, Optional, Tuple

import pytorch_lightning as ptl
import torch
from pytorch_lightning.cli import LightningCLI
from model.attentive_rnn import AttentiveRNN
from model.modeling_lina import LinaModel
from torch import nn
from transformers import get_cosine_schedule_with_warmup
from model.accuracy import MulticlassAccuracy

class TrainLina(ptl.LightningModule):
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
        txt_encoder: Optional[nn.Module] = None,
        spk_encoder: Optional[nn.Module] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        n_warmup_steps: int = 500,
        n_training_steps: int = 300000,
        mask_text_p: float = 0.,
        load_weights: Optional[str] = None,
    ):
        super(TrainLina, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps

        self.model = LinaModel(
                attentive_rnn,
                d_model,
                len(quant_layer),
                n_codebook,
                n_special_token_in,
                n_special_token_out,
                n_txt_vocab,
                tie_embed=tie_embed,
                txt_encoder=txt_encoder,
                spk_encoder=spk_encoder,
                mask_text_p=mask_text_p,
                )
        
        self.save_hyperparameters()

        self.accuracy_metric = MulticlassAccuracy(
            n_codebook + n_special_token_out,
            top_k=10,
            ignore_index=[0, 1],
        )
        if load_weights is not None:
            model = torch.load(load_weights)
            self.load_state_dict(model["state_dict"])


    def on_train_epoch_start(self):
        if hasattr(self.trainer.train_dataloader.batch_sampler, "set_epoch"):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)


    def step(self, batch):
        text_token = batch["text_token"]
        audio_token = batch["audio_token"]
        crossatt_mask = batch["crossatt_mask"]
        crossatt_pos = batch["crossatt_pos"]
        encoder_mask = batch["encoder_mask"]
        y_mask = batch["y_mask"]

        logits, loss, att, masked_logits, masked_target = self.model(text_token, audio_token, encoder_mask, crossatt_mask, logits_mask=y_mask, crossatt_pos=crossatt_pos)
        
        n_quant = masked_logits.shape[1]
        
        accs = []

        return logits, loss, att, accs

    def training_step(self, batch, idx):
        logits, loss, att, accs = self.step(batch)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for i, acc in enumerate(accs):
            self.log("train_acc_" + str(i), acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, idx):
        logits, loss, att, accs = self.step(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for i, acc in enumerate(accs):
            self.log("val_acc_" + str(i), acc, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        params = [
            {
                "params": self.model.parameters(),
                "weight_decay": self.weight_decay,
            }
        ]
        opt = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=self.betas,
        )
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps,)

        return [opt], [{'scheduler': scheduler, "interval": "step"}]

def cli(run=True):
    class LinaCli(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.init_args.quant_layer", "model.init_args.quant_layer"
            )

    return LinaCli(parser_kwargs={"parser_mode": "omegaconf"}, save_config_kwargs={"overwrite": True}, run=run)

if __name__ == "__main__":
    cli()
