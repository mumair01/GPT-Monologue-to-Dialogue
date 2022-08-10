# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 14:37:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-10 13:52:14

############################
# This script contains a data module for use with the re-implementation of TurnGPT.
############################

import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

from datasets import concatenate_datasets, load_from_disk, load_dataset
import pytorch_lightning as pl


# TODO: These cleanup methods are data specific (for the ICC data)
# and should not be here - dm should be data agnostic.

# def clean_speaker_labels(data):
#     """Remove speaker labels from the start and end of an utterance"""
#     if len(data['Utterance'].split()) > 1:
#         data['Utterance'] = " ".join(data['Utterance'].split()[1:-1])
#     return data


class TurnGPTFinetuneDM(pl.LightningDataModule):
    """
    Prepares Conversational data for finetuning TurnGPT.
    Expects csv files with dialogue where each turn is a different speaker
    and conversations have a unique identifier.
    """

    def __init__(self,
        tokenizer,
        train_csv_path : str,
        val_csv_path : str,
        conversation_id_key : str,
        utterance_key : str,
        save_dir : str,
        batch_size : int = 16,
        max_length : int = 1024,
        num_workers : int = 1,
        pin_memory : bool = True,
        num_proc = None,
        use_cache=False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.conversation_id_key = conversation_id_key
        self.utterance_key = utterance_key
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_proc = num_proc
        self.use_cache = use_cache

    ######################## OVERRIDDEN METHODS ##############################

    def prepare_data(self):
        dataset = load_dataset("csv", data_files={
                "train" : self.train_csv_path,
                "validation" : self.val_csv_path
            },
            download_mode="force_redownload" if self.use_cache else "reuse_cache_if_exists"
        )

        # Group the data based on conversations
        for split in ('train', 'validation'):
            dataset[split] = self._group_by(
                dataset[split],self.conversation_id_key,self._join)

        dataset = dataset.map(
            self._encode,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset["train"].column_names
        )
        dataset = dataset.map(
            self._chunk_tokenized_samples,
            batched=True
        )
        # Save the dataset
        dataset.set_format(type="torch")
        dataset.save_to_disk(self.save_dir)

    def setup(self, stage):
        if stage in (None, "fit"):
            dataset = load_from_disk(self.save_dir)
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
        else:
            raise NotImplementedError(
                f"Finetuning data module does not implement stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )
    ######################## ADDITIONAL METHODS ##############################

    def _encode(self, examples):
        toks = self.tokenizer(examples[self.utterance_key])
        return toks

    def _collate_fn(self, batch):
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["speaker_ids"] = self.tokenizer.pad(
            {"input_ids": [b["speaker_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        for k, v in ret.items():
            ret[k] = torch.tensor(v)
        return ret

    def _chunk_tokenized_samples(self, tokenized_samples, chunk_size=128):
        # Concatenate all the utterances
        keys =  ('input_ids','attention_mask','speaker_ids')
        concatenated_examples = {k : sum(tokenized_samples[k],[]) for k in keys}
        total_length = len(concatenated_examples[keys[0]])
        total_length = (total_length // chunk_size) * chunk_size
        chunks = {
            k : [concatenated_examples[k][i:i+ chunk_size] \
                for i in range(0, total_length,chunk_size)]  for k in keys
        }
        return chunks

    def _join(self, batch):
        return { self.utterance_key: [batch[self.utterance_key]]}

    def _group_by(self,d, col, join):
        """from: https://github.com/huggingface/datasets/issues/3644"""
        # Get the indices of each group
        groups = {key: [] for key in d.unique(col)}
        def create_groups_indices(key, i):
            groups[key].append(i)
        d.map(create_groups_indices, with_indices=True, input_columns=col)
        # Get one dataset object per group
        groups = {key: d.select(indices) for key, indices in groups.items()}
        # Apply join function
        groups = {
            key: dataset_group.map(join, batched=True, batch_size=len(dataset_group), remove_columns=d.column_names)
            for key, dataset_group in groups.items()
        }
        # Return concatenation of all the joined groups
        return concatenate_datasets(groups.values())

