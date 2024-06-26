from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from csv import reader
import pytorch_lightning as ptl
import torch
import math
import random
import bisect
from random import choices
import numpy as np
from typing import List, Optional, Tuple, Union, Dict
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

class BucketSampler(Sampler[List[int]]):
    def __init__(
            self,
            buckets: List[List[int]],
            batch_sizes: List[int] | int,
            bucket_sampling_weights: List[Tuple[float]] = None,
            drop_last: bool = True,
            distributed: bool = True,  # TODO - implement not distributed as well
            sample_bucket: Optional[int] = None,
            seed: int = 123,
            epoch_seed: bool = True,
            ):
        if type(batch_sizes) is int:
            batch_sizes = [batch_sizes] * len(buckets)
        else:
            assert len(buckets) == len(batch_sizes)

        if bucket_sampling_weights is not None:
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
        self.sample_bucket = sample_bucket
        self.epoch_seed = epoch_seed
        if self.sample_bucket is not None:
            self.total_sizes=[sample_bucket for bs in batch_sizes]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return sum(self.total_sizes)

    def __iter__(self):
        random.seed(self.seed + self.epoch * self.epoch_seed)
        for b in self.buckets:
            random.shuffle(b)
        buckets = self.buckets
        if self.sample_bucket is not None:
            buckets = [random.sample(b, bs*self.sample_bucket*self.num_replicas) for b, bs in zip(buckets, self.batch_sizes)]
        buckets = [b[self.rank :: self.num_replicas] for b in buckets]
        pool = [
                BatchSampler(SubsetRandomSampler(b), bs, drop_last=self.drop_last)
                for b, bs in zip(buckets, self.batch_sizes)
                ]
        pool = [iter(b) for b in pool]
        weights = (
                [w for w in self.bucket_sampling_weights]
                if self.bucket_sampling_weights is not None
                else None
                )
        while pool:  # sample until all buckets are done
            idx, bucket = choices(list(enumerate(pool)), weights=weights)[0]
            try:
                batch = next(bucket)
                yield batch
            except StopIteration:
                pool.pop(idx)  # if bucket is done, throw it
                if weights is not None:
                    weights.pop(idx)

def random_crop(al, dur):
    txt = [x[-1] for x in al]
    align = [(i, float(x[0])) for i, x in enumerate(al)]
    idx, start = zip(*align)
    pos = min(len(start)-1, bisect.bisect_left(start, dur)+1)
    stop_idx = idx[pos]
    dur = start[pos]
    txt = ",".join(txt[:stop_idx])
    txt = [1] + [int(x) for x in txt.split(",")] + [2]
    txt = torch.LongTensor(txt)

    return txt, dur

class LinaDataModule(ptl.LightningDataModule):
    def __init__(
            self,
            path,
            quant_layer,
            codec_rate_hz=75,
            batch_size=None,
            token_by_batch=None,
            num_workers=8,
            n_buckets=1,
            test_size=2000,
            seed=123,
            random_crop=False,
            bucket_size=None,
            min_len=None,
            max_len=None,
            ):
        super().__init__()
        self.path = path
        self.codec_rate_hz = codec_rate_hz
        self.batch_size = batch_size
        self.token_by_batch = token_by_batch
        self.num_workers = num_workers
        self.quant_layer = quant_layer
        self.n_buckets = n_buckets
        self.random_crop = random_crop
        self.bucket_size = bucket_size
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.test_size = test_size


    def setup(self, stage):
        self.dataset = load_dataset(self.path).with_format("torch").map(lambda x: {"len": x["audio_token"].shape[-1]}).filter(lambda x: x["align_token"] is not None and len(x["align_token"]) > 1)

        self.dataset = self.dataset["train"].train_test_split(test_size=self.test_size)

        train_lens, val_lens = map(lambda x: self.dataset[x]["len"].tolist(), ("train", "test"))
        minl = self.min_len if self.min_len is not None else min(train_lens)
        maxl = self.max_len if self.max_len is not None else max(train_lens)
        bound = np.linspace(minl, maxl+1, num=self.n_buckets+1)
        bound = [int(x) for x in bound]
        self.batch_bound = defaultdict(lambda: (minl, maxl))
        if self.token_by_batch is not None:
            for lb, hb in zip(bound[:-1], bound[1:]):
                self.batch_bound[self.token_by_batch//hb] = (lb, hb)
            def get_buckets(lens):
                def get_bucket_num(sz):
                    lb = bound[:-1]
                    hb = [maxl]*self.n_buckets if self.random_crop else bound[1:]
                    return [i for i, (low, high) in enumerate(zip(lb, hb)) if sz >= low and sz < high]
                buckets = defaultdict(lambda: [])
                for i, l in enumerate(lens):
                    for bn in get_bucket_num(l):
                        buckets[bn].append(i)
                buckets = [x for x in buckets.values()]
                return buckets
            batch_sizes = list(self.batch_bound.keys())
            train_buckets, val_buckets = map(get_buckets, (train_lens, val_lens))
            self.train_batch_sampler = BucketSampler(train_buckets, batch_sizes=batch_sizes, seed=self.seed, sample_bucket=self.bucket_size)
            self.val_batch_sampler = BucketSampler(val_buckets, batch_sizes=batch_sizes, seed=self.seed, epoch_seed=False)


    def collate_fn(self, batch):
        audio_token = [x["audio_token"] for x in batch]
        text_token = [x["text_token"] for x in batch]

        if self.random_crop:
            align_token = [x["align_token"] for x in batch]
            lb, hb = self.batch_bound[len(batch)]
            cut = random.randint(lb, hb)/self.codec_rate_hz
            text_token, dur = zip(*[random_crop(al, cut) for al in align_token])
            audio_token = [yy[..., :int(d * self.codec_rate_hz)] for yy, d in zip(audio_token, dur)]

        audio_token = [
                delay_rvq(
                    x.squeeze()[self.quant_layer] + 3,
                    head_token=1,
                    tail_token=2,
                    ).T
                for x in audio_token
                ]

        audio_lens, text_lens = map(lambda x: torch.LongTensor([len(t) for t in x]), (audio_token, text_token))
        audio_token, text_token = map(lambda x: pad_sequence(x, batch_first=True, padding_value=0), (audio_token, text_token))
        return text_token, audio_token, text_lens, audio_lens, None, None, None

    def train_dataloader(self):
        return DataLoader(
                self.dataset["train"],
                batch_size=1 if self.token_by_batch is not None else self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                batch_sampler=self.train_batch_sampler if self.token_by_batch is not None else None,
                )
    def val_dataloader(self):
        return DataLoader(
                self.dataset["test"],
                batch_size=1 if self.token_by_batch is not None else self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                batch_sampler=self.val_batch_sampler if self.token_by_batch is not None else None,
                )
