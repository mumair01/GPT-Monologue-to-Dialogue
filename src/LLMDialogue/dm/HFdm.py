# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-21 10:10:51
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-22 08:30:19

import sys
import os
from functools import partial
import pandas as pd
import torch

# Transformers
from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)

from typing import List, Dict, Any, Optional
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)
import pytorch_lightning as pl


class HFLanguageModelingDM(pl.LightningDataModule):

    def __init__(
        self,
        datasets: DatasetDict,
        tokenizer: PreTrainedTokenizer,
        save_dir: str,
        data_block_size: Optional[int] = 128,
        batched: Optional[bool] = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.data_block_size = data_block_size
        self.batched = batched
        self.save_dir = save_dir
        self.utterance_key = "text"
        self.data_collator = None
        # NOTE: Huggingface does not need this - does it automatically - adding
        # as arg for API consistency.
        self.device = device

    def prepare_data(self):

        # Now, concatenate these into one object.
        tokenized_datasets = self.datasets.map(
            self._tokenize_fn,
            batched=self.batched,
            remove_columns=self.datasets["train"].column_names,
        )
        # Chunking the tokenized data
        lm_datasets = tokenized_datasets.map(
            self._chunk_tokenized_samples,
            batched=self.batched,
        )
        # Create the data collator, which is responsible for creating batches from the
        # datasets during training.
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )
        # If output dir is specified, save the datasets to disk.
        os.makedirs(self.save_dir, exist_ok=True)
        lm_datasets.save_to_disk(self.save_dir)

    def setup(
        self,
        stage: Optional[str] = "fit",
    ):
        dataset = load_from_disk(self.save_dir)
        if stage in (None, "fit"):
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["validation"]
        elif stage == "test":
            self.test_dataset = dataset["test"]
        else:
            raise NotImplementedError(f"Data module does not implement stage: {stage}")

    def train_dataloader(self) -> Dataset:
        return self.train_dataset

    def val_dataloader(self) -> Dataset:
        return self.val_dataset

    def test_dataloader(self) -> Dataset:
        return self.test_dataset

    ########################
    # Private methods
    ########################

    def _tokenize_fn(self, data: Any):
        """
        Returns a method that takes in some data and tokenizes the utterance_key
        attribute of the data.
        """
        return self.tokenizer(data[self.utterance_key], truncation=True)

    # def _df_splits_to_datasets(
    #     self,
    #     df_map: Dict[str, pd.DataFrame],
    # ) -> DatasetDict:
    #     # NOTE: Each data frame must have a text field.
    #     dataset_map = {k: Dataset.from_pandas(v) for k, v in df_map.items()}
    #     return DatasetDict(dataset_map)

    def _chunk_tokenized_samples(
        self,
        tokenized_samples,
    ):
        """
        Given tokenized samples, each of some length L_i, chunks the samples into
        samples of size chunk_size.
        This makes sure that the data can fit in the model.
        """
        # Concatenate all the utterances
        keys = ("input_ids", "attention_mask")
        concatenated_examples = {k: sum(tokenized_samples[k], []) for k in keys}
        total_length = len(concatenated_examples[keys[0]])
        total_length = (total_length // self.data_block_size) * self.data_block_size
        chunks = {
            k: [
                concatenated_examples[k][i : i + self.data_block_size]
                for i in range(0, total_length, self.data_block_size)
            ]
            for k in keys
        }
        return chunks
