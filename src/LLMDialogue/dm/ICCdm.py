# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-21 11:43:49
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-24 12:47:45


import sys
import os
from functools import partial
import pandas as pd

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
from LLMDialogue.datasets import load
from LLMDialogue.datasets.ICC import ICCDataset, ICCDatasetSplits, objectify
from LLMDialogue.dm.utils import join, group_by, df_splits_to_datasets
from LLMDialogue.tokenizers import SpokenDialogueTokenizer
from .HFdm import HFLanguageModelingDM
from .spokenLanguageDM import SpokenLanguageModelingDM


class ICCHFLMDM(HFLanguageModelingDM):
    def __init__(
        self,
        dset_variant: str,
        tokenizer: PreTrainedTokenizer,
        save_dir: str,
        data_block_size: Optional[int] = 128,
        batched: Optional[bool] = True,
        group_by_conversation: Optional[str] = False,
        **dset_kwargs,
    ):

        self.dset_variant = dset_variant
        self.group_by_conversation = group_by_conversation
        super().__init__(
            self._create_datasets(**dset_kwargs),
            tokenizer,
            save_dir,
            data_block_size,
            batched,
        )

    # TODO: This wasn't done in the original experiment but I need to
    # determine if the dataset should be split by conversation - so that
    # each conversation itself is not shuffled.
    def _create_datasets(
        self,
        **dset_kwargs,
    ) -> DatasetDict:
        # Load the dataset with the specified configurations.
        dset = load(
            dset_name="ICC",
            config_name=self.dset_variant,
            **dset_kwargs,
        )
        dset_splits = objectify(dset)
        # Create a dataframe map from the dataset.
        df_map = dict()
        for name, df in dset_splits.data_df_map.items():
            remove_labels = [c for c in df.columns if c != "utterance"]
            # df.drop(labels=remove_labels)
            df = df.drop(remove_labels, axis=1)
            df = df.rename(columns={"utterance": "text"})
            if name == "test":
                name = "validation"
            df_map[name] = df
        # Convert the df map to datasets

        datasets = df_splits_to_datasets(df_map)
        # Now, group the data by conversation.
        if self.group_by_conversation:
            for split in df_map.keys():
                datasets[split] = group_by(
                    datasets[split], "conversationID", partial(join, key="text")
                )
        return datasets


class ICCSpokenLanguageModelingDM(SpokenLanguageModelingDM):

    def __init__(
        self,
        dset_variant: str,
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
        group_by_conversation: Optional[bool] = True,
        **dset_kwargs,
    ):
        self.dset_variant = dset_variant
        self.group_by_conversation = group_by_conversation
        super().__init__(
            self._create_datasets(**dset_kwargs),
            tokenizer,
            save_dir,
            batch_size,
            max_length,
            chunk_size,
            num_workers,
            pin_memory,
            num_proc,
            use_cache,
            batched,
        )

    def _create_datasets(
        self,
        **dset_kwargs,
    ) -> DatasetDict:
        # Load the dataset with the specified configurations.
        dset = load(
            dset_name="ICC",
            config_name=self.dset_variant,
            **dset_kwargs,
        )
        dset_splits = objectify(dset)
        # Create a dataframe map from the dataset.
        df_map = dict()
        for name, df in dset_splits.data_df_map.items():
            remove_labels = [
                c for c in df.columns if c != "utterance" and c != "conversationID"
            ]
            # df.drop(labels=remove_labels)
            df = df.drop(remove_labels, axis=1)
            df = df.rename(columns={"utterance": "text"})
            if name == "test":
                name = "validation"
            df_map[name] = df
        # Convert the df map to datasets
        datasets = df_splits_to_datasets(df_map)
        # Now, group the data by conversation.
        if self.group_by_conversation:
            for split in df_map.keys():
                datasets[split] = group_by(
                    datasets[split], "conversationID", partial(join, key="text")
                )
        return datasets
