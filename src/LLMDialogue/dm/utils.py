# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-21 11:35:44
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-21 11:37:19


import sys
import os
from functools import partial
import pandas as pd

# Transformers
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
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


def df_splits_to_datasets(df_map: Dict[str, pd.DataFrame]) -> DatasetDict:
    # NOTE: Each data frame must have a text field.
    dataset_map = {k: Dataset.from_pandas(v) for k, v in df_map.items()}
    return DatasetDict(dataset_map)


def group_by(d, col, join):
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
        key: dataset_group.map(
            join,
            batched=True,
            batch_size=len(dataset_group),
            remove_columns=d.column_names,
        )
        for key, dataset_group in groups.items()
    }
    # Return concatenation of all the joined groups
    return concatenate_datasets(list(groups.values()))


def join(batch, key: str):
    """Simply return the Utterance values of the joined batch."""
    return {key: [batch[key]]}
