import math
import os
import random
from pathlib import Path
from random import choices
from typing import List, Optional, Tuple

import polars as pl
import torch
from pytorch_lightning import LightningDataModule
from safetensors import safe_open
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    Sampler,
    SubsetRandomSampler,
)


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


def left_pad_sequence(seq, batch_first=False, **kwargs):
    seq_reversed = [torch.flip(x, dims=[0]) for x in seq]
    if batch_first:
        return torch.flip(
            pad_sequence(seq_reversed, batch_first=batch_first, **kwargs), dims=[1]
        )
    else:
        return torch.flip(
            pad_sequence(seq_reversed, batch_first=batch_first, **kwargs), dims=[0]
        )


def pad_2d_sequence(seq, padding_value=0):
    max_x, max_y = map(max, zip(*map(lambda x: x.shape, seq)))
    pad = lambda x: torch.nn.functional.pad(
        x,
        (0, max_y - x.shape[1], 0, max_x - x.shape[0]),
        value=padding_value,
    )
    return torch.stack([pad(x) for x in seq])

# From https://github.com/shivammehta25/Matcha-TTS
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
matcha_symbols = list(_punctuation) + list(_letters_ipa) + list(_letters)


def exists(x):
    return x is not None


class ManifestDataModule(LightningDataModule):
    def __init__(
        self,
        train_manifest_path: Path,
        root_path: Path,
        batch_size,
        val_manifest_path: Optional[Path] = None,
        env_root: Optional[str] = None,
        num_workers=16,
        val_dataset_size=1000,
        symbols_file: Optional[Path] = None,
        seed=123,
        bucket_by_quantile: Optional[int] = None,
        shuffle: bool = False,
        code_col=".safetensors",
        phon_col=".phonemized.txt",
        metadata_col=None,
        filter_min=None,
        filter_max=None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_dataset_size = val_dataset_size

        if env_root is not None:
            root_path = Path(os.environ[env_root]) / root_path
            train_manifest_path = Path(os.environ[env_root]) / train_manifest_path
            val_manifest_path = (
                Path(os.environ[env_root]) / val_manifest_path
                if exists(val_manifest_path)
                else None
            )
        self.root_path = root_path
        self.seed = seed
        self.train_manifest_path = train_manifest_path
        self.val_manifest_path = val_manifest_path
        self.dataset_kwargs = dataset_kwargs
        self.bucket_by_quantile = bucket_by_quantile
        if exists(symbols_file):
            with open(symbols_file, "r") as f:
                f = f.read()
            self.symbols = list(eval(f))
            print("SYMBOLS: ", self.symbols)
        else:
            self.symbols = None

        self.metadata_col = metadata_col
        self.code_col = code_col
        self.phon_col = phon_col
        self.filter_min = filter_min
        self.filter_max = filter_max
        self.shuffle = shuffle

    def setup(self, stage: str):
        def manifest_to_list(json, bucket=None):
            if exists(self.filter_min):
                json = json.filter(pl.col("length") > self.filter_min)
            if exists(self.filter_max):
                json = json.filter(pl.col("length") < self.filter_max)
            if exists(bucket):
                buckets = (
                    json.with_row_count()
                    .select(["row_nr", "length"])
                    .with_columns(
                        pl.col("length")
                        .qcut(
                            [
                                i / self.bucket_by_quantile
                                for i in range(self.bucket_by_quantile)
                            ]
                        )
                        .alias("qlength")
                    )
                    .partition_by("qlength")
                )
                buckets = [x["row_nr"].to_list() for x in buckets]
                self.batch_sampler = BucketSampler(
                    buckets,
                    batch_sizes=self.batch_size,
                    seed=self.seed,
                )
            else:
                self.batch_sampler = None

            phon_list = json[self.phon_col].to_list()
            code_list = json[self.code_col].to_list()
            if exists(self.metadata_col):
                metadata_list = json[self.metadata_col].to_dict(as_series=False)
                metadata_list = [
                    {k: v for k, v in zip(metadata_list.keys(), x)}
                    for x in zip(*metadata_list.values())
                ]
            else:
                metadata_list = None
            phon_list = [self.root_path / p for p in phon_list]
            code_list = [self.root_path / p for p in code_list]
            return phon_list, code_list, metadata_list

        if exists(self.val_manifest_path):
            val_manifest = pl.read_json(self.val_manifest_path)
            train_manifest = pl.read_json(self.train_manifest_path)
        else:
            manifest = pl.read_json(self.train_manifest_path)
            manifest = manifest.sample(
                fraction=1.0, seed=self.seed, shuffle=self.shuffle
            )
            val_manifest, train_manifest = manifest.head(
                self.val_dataset_size
            ), manifest.tail(-self.val_dataset_size)

        val_phon, val_code, val_meta = manifest_to_list(val_manifest)
        train_phon, train_code, train_meta = manifest_to_list(
            train_manifest, bucket=self.bucket_by_quantile
        )
        self.train_dataset = ManifestDataset(
            train_phon,
            train_code,
            metadata=train_meta,
            symbols=self.symbols,
            **self.dataset_kwargs,
        )
        self.val_dataset = ManifestDataset(
            val_phon,
            val_code,
            metadata=val_meta,
            symbols=self.symbols,
            **self.dataset_kwargs,
        )
        self.collate_fn = self.train_dataset.collate_fn

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=1
            if exists(self.bucket_by_quantile)
            else self.batch_size,  # 1 "means None"
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_sampler=self.batch_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class BucketSampler(Sampler[List[int]]):
    def __init__(
        self,
        buckets: List[List[int]],
        batch_sizes: List[int] | int,
        bucket_sampling_weights: List[Tuple[float]] = None,
        drop_last: bool = True,
        distributed: bool = True,  # TODO - implement not distributed as well
        seed: int = 123,
    ):
        if type(batch_sizes) is int:
            batch_sizes = [batch_sizes] * len(buckets)
        else:
            assert len(buckets) == len(batch_sizes)

        if exists(bucket_sampling_weights):
            assert len(bucket_sampling_weights) == len(batch_sizes)
        self.bucket_sampling_weights = bucket_sampling_weights
        self.buckets = buckets
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.num_samples = [
            math.ceil((len(b) - self.num_replicas) / self.num_replicas) for b in buckets
        ]
        self.batch_sizes = batch_sizes
        self.total_sizes = [
            ns // bs for ns, bs in zip(self.num_samples, self.batch_sizes)
        ]
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return sum(self.total_sizes)

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        for b in self.buckets:
            random.shuffle(b)
        buckets = [b[self.rank :: self.num_replicas] for b in self.buckets]
        pool = [
            BatchSampler(SubsetRandomSampler(b), bs, drop_last=self.drop_last)
            for b, bs in zip(buckets, self.batch_sizes)
        ]
        pool = [iter(b) for b in pool]
        weights = (
            [w for w in self.bucket_sampling_weights]
            if exists(self.bucket_sampling_weights)
            else None
        )
        while pool:  # sample until all buckets are done
            idx, bucket = choices(list(enumerate(pool)), weights=weights)[0]
            try:
                batch = next(bucket)
                yield batch
            except StopIteration:
                pool.pop(idx)  # if bucket is done, throw it
                if exists(weights):
                    weights.pop(idx)


class ManifestDataset(Dataset):
    def __init__(
        self,
        phon,
        code,
        quant_layer,
        codebook_size: int = 2**10,
        metadata: Optional[List] = None,
        symbols: Optional[str] = None,
    ):
        super().__init__()

        self.phon = phon
        self.code = code
        self.metadata = metadata if exists(metadata) else list(range(len(phon)))

        self.symbols = symbols if exists(symbols) else matcha_symbols
        self.text_vocab_size = len(self.symbols)
        self.special_symbols_in = ["BOS", "EOS"]
        self.special_symbols_out = ["BOA", "EOA"]
        if type(quant_layer) is int:
            self.quant_layer = [quant_layer]
        else:
            self.quant_layer = quant_layer

        self.vocab_x = ["PAD"] + self.special_symbols_in + self.symbols
        self.vocab_y = ["PAD"] + self.special_symbols_out + list(range(codebook_size))

        self.vocab_x_code = {x: i for i, x in enumerate(self.vocab_x)}
        self.vocab_y_code = {x: i for i, x in enumerate(self.vocab_y)}

        self.vocab_x_decode = {i: x for i, x in enumerate(self.vocab_x)}
        self.vocab_y_decode = {i: x for i, x in enumerate(self.vocab_y)}

    def prepare_tensors(self, code, phon):
        phon = phon.strip()
        x = [self.vocab_x_code[p] for p in phon]
        x = torch.LongTensor(
            [self.vocab_x_code["BOS"]] + x + [self.vocab_x_code["EOS"]]
        )
        y = delay_rvq(
            code[self.quant_layer, :] + len(self.special_symbols_in),
            head_token=self.vocab_y_code["BOA"],
            tail_token=self.vocab_y_code["EOA"],
        ).T

        return x, y, code, phon

    def load(
        self,
        code_path,
        phon_path,
    ):
        # code = torch.load(code_path, map_location="cpu").squeeze(0)
        with safe_open(code_path, framework="pt", device="cpu") as f:
            code = f.get_tensor("code").squeeze()
        with open(phon_path, "r") as f:
            phon = f.read()
        return code, phon

    def __getitem__(self, idx):
        return (
            *self.prepare_tensors(*self.load(self.code[idx], self.phon[idx])),
            self.metadata[idx],
        )

    def __len__(self):
        return len(self.code)

    def collate_fn(self, batch: List[Tensor]):
        x, y, code, phon, metadata = zip(*batch)
        x_len = torch.LongTensor([len(t) for t in x])
        y_len = torch.LongTensor([len(t) for t in y])
        x = pad_sequence(x, batch_first=True, padding_value=self.vocab_x_code["PAD"])
        y = pad_sequence(y, batch_first=True, padding_value=self.vocab_y_code["PAD"])

        return (
            x,
            y,
            x_len,
            y_len,
            code,
            phon,
            metadata,
        )


class ManifestRWKVDataset(Dataset):
    def __init__(
        self,
        phon,
        code,
        quant_layer,
        metadata: Optional[List] = None,
        quant_bias_shift: int = 0,
        quant_level_shift: int = 1024,
        symbols=None,
    ):
        super().__init__()

        self.phon = phon
        self.code = code
        self.metadata = metadata if exists(metadata) else list(range(len(phon)))

        self.symbols = symbols if exists(symbols) else matcha_symbols
        self.text_vocab_size = len(self.symbols)
        self.special_symbols_in = ["BOS", "EOS"]
        self.special_symbols_out = ["BOA", "EOA"]
        self.encodec_symbols = []
        if type(quant_layer) is int:
            self.quant_layer = [quant_layer]
        else:
            self.quant_layer = quant_layer

        self.quant_level_shift = quant_level_shift
        self.quant_bias_shift = quant_bias_shift

        for q in self.quant_layer:
            for i in range(self.quant_level_shift):
                self.encodec_symbols.append("Q" + str(q) + "|" + str(i))

        self.vocab_x = ["PAD"] + self.special_symbols_in + self.symbols
        self.vocab_y = ["PAD"] + self.special_symbols_out + self.encodec_symbols

        self.vocab_x_code = {x: i for i, x in enumerate(self.vocab_x)}
        self.vocab_y_code = {x: i for i, x in enumerate(self.vocab_y)}

        self.vocab_x_decode = {i: x for i, x in enumerate(self.vocab_x)}
        self.vocab_y_decode = {i: x for i, x in enumerate(self.vocab_y)}

    def prepare_tensors(self, code, phon):
        shift = (
            torch.arange(len(self.quant_layer)) * self.quant_level_shift
            + self.quant_bias_shift
        ).unsqueeze(1)
        phon = phon.strip()
        x = [self.vocab_x_code[p] for p in phon]
        x = torch.LongTensor(
            [self.vocab_x_code["BOS"]] + x + [self.vocab_x_code["EOS"]]
        )
        y = delay_rvq(
            code[self.quant_layer, :] + shift + len(self.special_symbols_in),
            head_token=self.vocab_y_code["BOA"],
            tail_token=self.vocab_y_code["EOA"],
        ).T

        return x, y, code, phon

    def load(
        self,
        code_path,
        phon_path,
    ):
        # code = torch.load(code_path, map_location="cpu").squeeze(0)
        with safe_open(code_path, framework="pt", device="cpu") as f:
            code = f.get_tensor("code").squeeze()
        with open(phon_path, "r") as f:
            phon = f.read()
        return code, phon

    def __getitem__(self, idx):
        return (
            *self.prepare_tensors(*self.load(self.code[idx], self.phon[idx])),
            self.metadata[idx],
        )

    def __len__(self):
        return len(self.code)

    def collate_fn(self, batch: List[Tensor]):
        x, y, code, phon, metadata = zip(*batch)
        x_len = torch.LongTensor([len(t) for t in x])
        y_len = torch.LongTensor([len(t) for t in y])
        x = pad_sequence(x, batch_first=True, padding_value=self.vocab_x_code["PAD"])
        y = pad_sequence(y, batch_first=True, padding_value=self.vocab_y_code["PAD"])

        return (
            x,
            y,
            x_len,
            y_len,
            code,
            phon,
            metadata,
        )


if __name__ == "__main__":
    ds = list(range(100))
    cuts = sorted(choices(list(range(100)), k=4))
    print(cuts)
    ds_cuts = [ds[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    print(ds_cuts)
    bs = BucketSampler(ds_cuts, [3, 4, 5], drop_last=False)
    print("sampling")
    for b in bs:
        print(b)
