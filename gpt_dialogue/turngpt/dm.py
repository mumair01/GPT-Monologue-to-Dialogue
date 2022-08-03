# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 14:37:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-03 15:48:39

############################
# This script contains a data module for use with the re-implementation of TurnGPT
# and is inspired by Erik's datasets_turntaking repo.
# NOTE: This code is taken from the repository specified below and may or may
# not be modified.
# Acknowledgements:
#   Paper: https://arxiv.org/abs/2010.10874
#   Code: https://github.com/ErikEkstedt/datasets_turntaking/tree/main/datasets_turntaking
############################

import torch
import os
import sys
from torch.utils.data import DataLoader

from datasets import concatenate_datasets, load_from_disk, load_dataset
import pytorch_lightning as pl


# TODO: Eventually, don't hard code this.
_ROOT_PATH = '/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue'
_ICC_PATHS = {
    'train' : os.path.join(_ROOT_PATH,'data/datasets/processed/ICC/julia_dissertation/train.csv'),
    'validation' : os.path.join(_ROOT_PATH,'data/datasets/processed/ICC/julia_dissertation/validation.csv'),
    'test' : os.path.join(_ROOT_PATH,'data/datasets/processed/ICC/julia_dissertation/test.csv')
}
_SETUP_SAVE_DIR = os.path.join(_ROOT_PATH,'data/turngpt_dm_test')


# TODO: What should I do about the conversation start and end tokens?
class TurnGPTDM(pl.LightningDataModule):

    _DATASETS = ("icc")

    def __init__(self,
            tokenizer,
            dataset,
            batch_size=16,
            max_length=1024,
            num_workers=1,
            pin_memory=True,
            num_proc=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_proc = num_proc


    ######################## ADDITIONAL METHODS ##############################

    def clean_speaker_labels(self,data):
        if len(data['Utterance'].split()) > 1:
            data['Utterance'] = " ".join(data['Utterance'].split()[1:-1])
        return data

    def encode(self, examples):
        toks = self.tokenizer(examples["Utterance"])
        return toks
        # return {"input_ids": toks["input_ids"], "speaker_ids": toks["speaker_ids"]}

    def collate_fn(self, batch):
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["speaker_ids"] = self.tokenizer.pad(
            {"input_ids": [b["speaker_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        for k, v in ret.items():
            ret[k] = torch.tensor(v)
        return ret

    def chunk_tokenized_samples(self, tokenized_samples,chunk_size=128):
        # Concatenate all the utterances
        keys =  ('input_ids','attention_mask','speaker_ids')
        concatenated_examples = {k : sum([tokenized_samples[k]],[]) for k in keys}
        total_length = len(concatenated_examples[keys[0]])
        total_length = (total_length // chunk_size) * chunk_size
        chunks = {
            k : [concatenated_examples[k][i:i+ chunk_size] \
                for i in range(0, total_length,chunk_size)]  for k in keys
        }
        return chunks


    ######################## OVERRIDDEN METHODS ##############################

    # TODO: There is a hashing issue here with map for some reason.
    def prepare_data(self):
        dataset = load_dataset("csv", data_files=_ICC_PATHS)
        # Clean the speaker labels i.e., remove them from the data.
        dataset = dataset.map(
            self.clean_speaker_labels,
            batched=False,
            remove_columns=["Unnamed: 0","convName","convID"]
        )
        dataset = dataset.map(
            self.encode,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset["train"].column_names
        )
        # Chunk the data
        dataset = dataset.map(
            self.chunk_tokenized_samples,
            batched=True)

        # Tokenize the dataset as well
        dataset.set_format(type="torch")
        dataset.save_to_disk(_SETUP_SAVE_DIR)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            dataset = load_from_disk(_SETUP_SAVE_DIR)
            self.train_dataset = dataset['train']
            self.validation_dataset = dataset['validation']
        if stage == "fit":
            dataset = load_from_disk(_SETUP_SAVE_DIR)
            self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        pass

