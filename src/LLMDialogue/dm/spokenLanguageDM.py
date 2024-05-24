# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-21 11:20:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-24 12:46:22

import torch
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)
from LLMDialogue.tokenizers import SpokenDialogueTokenizer

from datasets import concatenate_datasets, load_from_disk, load_dataset
import pytorch_lightning as pl
from LLMDialogue.utils.general import get_system_device


class SpokenLanguageModelingDM(pl.LightningDataModule):

    def __init__(
        self,
        datasets: DatasetDict,
        tokenizer: SpokenDialogueTokenizer,
        save_dir: str,
        batch_size: int = 16,
        max_length: int = 1024,
        chunk_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = True,
        num_proc=None,
        use_cache=False,
        batched: Optional[bool] = True,
        device: Optional[torch.device] = None,
    ):
        assert isinstance(tokenizer, SpokenDialogueTokenizer)
        super().__init__()
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_proc = num_proc
        self.use_cache = use_cache
        self.batched = batched
        self.utterance_key = "text"

        if device is None:
            device = get_system_device()
        self.device = device

    # TODO: Potentially add a method to be able to pass kwargs to the
    # tokenizer encode method.
    def prepare_data(self):
        datasets = self.datasets.map(
            self._encode,
            batched=self.batched,
            num_proc=self.num_proc,
            remove_columns=self.datasets["train"].column_names,
        )
        datasets = datasets.map(self._chunk_tokenized_samples, batched=True)
        # Save the dataset
        datasets.set_format(type="torch")
        datasets.save_to_disk(self.save_dir)

    def setup(
        self,
        stage: Optional[str] = "fit",
    ):
        dataset = load_from_disk(self.save_dir)
        if stage in (None, "fit"):
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["validation"]
        elif stage == "test":
            self.val_dataset = dataset["test"]
        else:
            raise NotImplementedError(f"Data module does not implement stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )

    def _encode(self, examples):
        toks = self.tokenizer(examples[self.utterance_key])
        return toks

    def _collate_fn(self, batch):
        """
        Responsible for preparing a batch for iteration.
        Here, we pad both the input and speaker ids and ensure the batch
        size does not increase self.max_length
        Additionally, moves the tensors to the appropriate device.
        """
        # Pad all the input ids per batch.
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["speaker_ids"] = self.tokenizer.pad(
            {"input_ids": [b["speaker_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        for k, v in ret.items():
            # TODO: Make sure that this requires_grad should be True.
            ret[k] = v.clone().detach().requires_grad_(False)
        return ret

    def _chunk_tokenized_samples(self, tokenized_samples):
        """Chunk the given tokenized samples based on the size."""
        if self.chunk_size > self.max_length:
            print(
                f"WARNING: Chunk size {self.chunk_size} greater than max length "
                f"{self.max_length} may lead to data loss"
            )
        # Concatenate all the utterances
        keys = ("input_ids", "attention_mask", "speaker_ids")
        concatenated_examples = {k: sum(tokenized_samples[k], []) for k in keys}
        total_length = len(concatenated_examples[keys[0]])
        total_length = (total_length // self.chunk_size) * self.chunk_size
        chunks = {
            k: [
                concatenated_examples[k][i : i + self.chunk_size]
                for i in range(0, total_length, self.chunk_size)
            ]
            for k in keys
        }
        return chunks
