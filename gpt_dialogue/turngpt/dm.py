# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 14:37:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-29 09:59:20

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

from gpt_dialogue.turngpt.tokenizer import SpokenDialogTokenizer

# TODO: Eventually, don't hard code this.
_ROOT_PATH = '/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue'
_ICC_PATHS = {
    'train' : os.path.join(_ROOT_PATH,'data/datasets/processed/ICC/julia_dissertation/train.csv'),
    'validation' : os.path.join(_ROOT_PATH,'data/datasets/processed/ICC/julia_dissertation/validation.csv'),
    'test' : os.path.join(_ROOT_PATH,'data/datasets/processed/ICC/julia_dissertation/test.csv')
}
_SETUP_SAVE_DIR = os.path.join(_ROOT_PATH,'data/turngpt_dm_test')


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
        t = self.tokenizer(examples["dialog"])
        return {"input_ids": t["input_ids"], "speaker_ids": t["speaker_ids"]}

    def collate_fn(self, batch):
        print(batch)
        sys.exit(-1)

    ######################## OVERRIDDEN METHODS ##############################

    # TODO: There is a hashing issue here with map for some reason.
    def prepare_data(self):
        dataset = load_dataset("csv", data_files=_ICC_PATHS)
        dataset = dataset.map(
            self.clean_speaker_labels,
            batched=False,
            remove_columns=["Unnamed: 0","convName","convID"]
        )
        # Tokenize the dataset as well
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

if __name__ == "__main__":
    TOKENIZER_CHECKPOINT = "gpt2"
    TRAIN_PATH = ""
    # Load tokenizer
    tokenizer = SpokenDialogTokenizer(TOKENIZER_CHECKPOINT)

    dm = TurnGPTDM(
        tokenizer=tokenizer,
        dataset="icc",
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    print(batch)
