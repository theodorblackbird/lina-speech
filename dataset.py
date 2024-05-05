from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as ptl
import torch

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

   
class LinaDataModule(ptl.LightningDataModule):
    def __init__(self, path, batch_size, quant_layer, num_workers=8):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.quant_layer = quant_layer

    def setup(self, stage):
        self.dataset = load_dataset(self.path).with_format("torch")

    def collate_fn(self, batch):
        audio_token = [
                    delay_rvq(
                        x["audio_token"].squeeze()[self.quant_layer] + 3,
                        head_token=1,
                        tail_token=2,
                        ).T
                    for x in batch
                ]

        text_token = [x["text_token"] for x in batch]
        audio_lens, text_lens = map(lambda x: torch.LongTensor([len(t) for t in x]), (audio_token, text_token))
        audio_token, text_token = map(lambda x: pad_sequence(x, batch_first=True, padding_value=0), (audio_token, text_token))
        return text_token, audio_token, text_lens, audio_lens, None, None, None
 
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)
