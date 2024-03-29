# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-11 15:55:19
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-17 16:48:23

import sys
import os
from functools import partial

# Transformers
from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)
from datasets import load_dataset


def tokenize_fn(tokenizer, utterance_key: str):
    """
    Returns a method that takes in some data and tokenizes the utterance_key
    attribute of the data.
    """
    return lambda data: tokenizer(data[utterance_key], truncation=True)


def chunk_tokenized_samples(tokenized_samples, chunk_size: int):
    """
    Given tokenized samples, each of some length L_i, chunks the samples into
    samples of size chunk_size.
    This makes sure that the data can fit in the model.
    """
    # Concatenate all the utterances
    keys = ("input_ids", "attention_mask")
    concatenated_examples = {k: sum(tokenized_samples[k], []) for k in keys}
    total_length = len(concatenated_examples[keys[0]])
    total_length = (total_length // chunk_size) * chunk_size
    chunks = {
        k: [
            concatenated_examples[k][i : i + chunk_size]
            for i in range(0, total_length, chunk_size)
        ]
        for k in keys
    }
    return chunks


class GPT2FinetuneDM:
    """
    Data module that can be used to prepare a data collator and datasets that
    can be used with a transformers Trainer object.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        train_csv_path: str,
        val_csv_path: str,
        utterance_key: str,
        data_block_size: int = 128,
    ):
        """_summary_

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            Tokenizer to use to generate the data - must be pretrained for gpt2.
        train_csv_path : str
            Path to the training csv file
        val_csv_path : str
            Path to the validation csv file.
        utterance_key : str
            Name of the column that contains the utterance id in the csv.
        data_block_size : int, optional
            Chunk size of the data in the collator, by default 128
        """
        self.tokenizer = tokenizer
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.utterance_key = utterance_key
        self.data_block_size = data_block_size

    def __call__(self):
        """
        Applies the transforms on the data and returns the loaded datasets

        Returns
        -------
        Tuple
            Returns a tuple of two items:
                1. Data collator for Trainer object.
                2. Tokenized training and validation datasets that can directly
                    be used by the Trainer object.
        """
        dataset = load_dataset(
            "csv",
            data_files={
                "train": self.train_csv_path,
                "validation": self.val_csv_path,
            },
        )

        # Once loaded, the dataset needs to be processed
        tokenized_datasets = dataset.map(
            tokenize_fn(self.tokenizer, self.utterance_key),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Chunking the tokenized data
        lm_datasets = tokenized_datasets.map(
            partial(chunk_tokenized_samples, chunk_size=self.data_block_size),
            batched=True,
        )

        # Create the data collator, which is responsible for creating batches from the
        # datasets during training.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )

        return data_collator, lm_datasets
