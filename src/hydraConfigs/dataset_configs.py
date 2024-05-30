# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-29 17:06:13
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 12:25:47

import sys
import os
from hydra_zen import hydrated_dataclass
from typing import List, Dict, Tuple, Any
from LLMDialogue.baseConfigs import BaseConfigs


##############
# GLOBALS
##############

ICC_DATA_FILE_PATHS = {
    "28_train_14_test": {
        "train": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/28_train_14_test_set/train"
        ),
        "test": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/28_train_14_test_set/test"
        ),
    },
    "5_train_37_test": {
        "train": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/5_train_37_test_set/train"
        ),
        "test": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/5_train_37_test_set/test"
        ),
    },
}


SI_DATASET_PATH = os.path.join(BaseConfigs.root_path, "data/raw/speaker_identity_stims")

##############
# BASIC CONFIGS
##############


@hydrated_dataclass(target=Dict)
class BaseDatasetConfig:
    version: str = "0.0.1"
    data_dir: str = None
    seed: int = None


##############
# ICC DATASET CONFIGS
##############


@hydrated_dataclass(target=Dict)
class ICCUnlabeledDatasetConfig(BaseDatasetConfig):
    variant_name: str = "no_labels"
    add_eos_tok_bw_turns = False
    additional_special_tokens: Dict = {
        "eos_token": "<ts>",
    }


@hydrated_dataclass(target=Dict)
class ICCLabeledDatasetConfig(BaseDatasetConfig):
    variant_name: str = "labels"
    additional_special_tokens: Dict = {
        "sp_start_token": "<START>",
        "sp_end_token": "<END>",
        "eos_token": "|<endoftext>|",
    }


@hydrated_dataclass(target=Dict)
class ICCLabeled5Train37TestConfig(ICCLabeledDatasetConfig):
    data_files: Dict = ICC_DATA_FILE_PATHS["5_train_37_test"]


@hydrated_dataclass(target=Dict)
class ICCLabeled28Train14TestConfig(ICCLabeledDatasetConfig):
    data_files: Dict = ICC_DATA_FILE_PATHS["28_train_14_test"]


@hydrated_dataclass(target=Dict)
class ICCUnlabeled5Train37TestConfig(ICCUnlabeledDatasetConfig):
    data_files: Dict = ICC_DATA_FILE_PATHS["5_train_37_test"]


@hydrated_dataclass(target=Dict)
class ICCUnlabeled28Train14TestConfig(ICCUnlabeledDatasetConfig):
    data_files: Dict = ICC_DATA_FILE_PATHS["28_train_14_test"]


##############
# SPEAKER IDENTITY DATASET CONFIGS
##############


@hydrated_dataclass(target=Dict)
class SpeakerIdentityDatasetConfigs(BaseConfigs):
    data_dir: str = SI_DATASET_PATH


@hydrated_dataclass(target=Dict)
class SIDatasetLabeledConfigs(SpeakerIdentityDatasetConfigs):
    variant_name: str = "labels"
    additional_special_tokens: Dict = (
        {
            "sp_start_token": "<START>",
            "sp_end_token": "<END>",
            "eos_token": "|<endoftext>|",
        },
    )


@hydrated_dataclass(target=Dict)
class SIDatasetUnlabeledConfigs(BaseDatasetConfig):
    variant_name: str = "no_labels"
    additional_special_tokens: Dict = (
        {
            "eos_token": "<ts>",
        },
    )
