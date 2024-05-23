# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 15:44:55
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 17:38:44


import sys
import os
from copy import deepcopy
from LLMDialogue.baseConfigs import BaseConfigs


MISSING = None
GLOBAL_SEED = 42

##################
# TRACKERS
##################

WANDB_CONFIGS = {
    "project": "gpt_monologue_to_dialogue",
    "entity": "hilab",
    "mode": "online",
    # "run_name": "test",
    "log_model": "end",
}


##################
# DATASETS
##################

DSET_VARIANTS = ("labels", "no_labels")

DATA_FILE_PATHS = {
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


ICCDATASET = {
    "version": "0.0.1",
    "seed": GLOBAL_SEED,
}

ICCDATASET_NO_LABELS = deepcopy(ICCDATASET)
ICCDATASET_NO_LABELS.update(
    {
        "variant_name": "no_labels",
        "additional_special_tokens": {
            "eos_token": "<ts>",
        },
    }
)


ICCDATASET_LABELS = deepcopy(ICCDATASET)
ICCDATASET_LABELS.update(
    {
        # "variant_name": "labels",
        "additional_special_tokens": {
            "sp_start_token": "<START>",
            "sp_end_token": "<END>",
        },
    },
)


ICCDATASET_LABELS_5_TRAIN_37_TEST = deepcopy(ICCDATASET_LABELS)
ICCDATASET_LABELS_5_TRAIN_37_TEST.update(
    {"data_files": DATA_FILE_PATHS["5_train_37_test"]}
)

ICCDATASET_LABELS_28_TRAIN_14_TEST = deepcopy(ICCDATASET_LABELS)
ICCDATASET_LABELS_28_TRAIN_14_TEST.update(
    {"data_files": DATA_FILE_PATHS["28_train_14_test"]}
)

ICCDATASET_NO_LABELS_5_TRAIN_37_TEST = deepcopy(ICCDATASET_NO_LABELS)
ICCDATASET_NO_LABELS_5_TRAIN_37_TEST.update(
    {"data_files": DATA_FILE_PATHS["5_train_37_test"]}
)

ICCDATASET_NO_LABELS_28_TRAIN_14_TEST = deepcopy(ICCDATASET_NO_LABELS)
ICCDATASET_NO_LABELS_28_TRAIN_14_TEST.update(
    {"data_files": DATA_FILE_PATHS["28_train_14_test"]}
)

SD_DATASET = {
    "variant_name": "labels",
    "version": "0.0.1",
    "seed": GLOBAL_SEED,
    "data_dir": os.path.join(BaseConfigs.root_path, "data/raw/speaker_identity_stims"),
}
SD_DATASET_NO_LABELS = deepcopy(SD_DATASET)
SD_DATASET_NO_LABELS.update(
    {
        "additional_special_tokens": {
            "eos_token": "<ts>",
        }
    }
)
SD_DATASET_LABELS = deepcopy(SD_DATASET)
SD_DATASET_LABELS.update(
    {
        "additional_special_tokens": {
            "sp_start_token": "<START>",
            "sp_end_token": "<END>",
        }
    }
)
##################
# DATAMODULES
##################


ICCHFLMDM_CONFIGS = {
    "dset_variant": MISSING,
    "tokenizer": MISSING,
    "save_dir": MISSING,
    "data_block_size": 128,
    "batched": True,
}

# NOTE: Dset variant, tokenizer, and save_dir need to be specified later.
ICCSpokenLanguageModelingDM_CONFIGS = {
    "dset_variant": MISSING,
    "tokenizer": MISSING,
    "save_dir": MISSING,
    "batch_size": 16,
    "max_length": 1024,
    "chunk_size": 128,
    "num_workers": 1,
    "pin_memory": True,
    "num_proc": None,
    "use_cache": False,
    "batched": True,
}

ICCHFLMDM_LABELS = deepcopy(ICCHFLMDM_CONFIGS)
ICCHFLMDM_LABELS.update({"dset_variant": "labels"})

ICCHFLMDM_LABELS_5_TRAIN_37_TEST = deepcopy(ICCHFLMDM_LABELS)
ICCHFLMDM_LABELS_5_TRAIN_37_TEST.update(deepcopy(ICCDATASET_LABELS_5_TRAIN_37_TEST))

ICCHFLMDM_LABELS_28_TRAIN_14_TEST = deepcopy(ICCHFLMDM_LABELS)
ICCHFLMDM_LABELS_28_TRAIN_14_TEST.update(deepcopy(ICCDATASET_LABELS_28_TRAIN_14_TEST))


ICCSpokenLanguageModelingDM_NO_LABELS = deepcopy(ICCSpokenLanguageModelingDM_CONFIGS)
ICCSpokenLanguageModelingDM_NO_LABELS.update({"dset_variant": "no_labels"})


ICCSpokenLanguageModelingDM_NO_LABELS_5_TRAIN_37_TEST = deepcopy(
    ICCSpokenLanguageModelingDM_NO_LABELS
)
ICCSpokenLanguageModelingDM_NO_LABELS_5_TRAIN_37_TEST.update(
    deepcopy(ICCSpokenLanguageModelingDM_NO_LABELS_5_TRAIN_37_TEST)
)

ICCSpokenLanguageModelingDM_NO_LABELS_28_TRAIN_14_TEST = deepcopy(
    ICCSpokenLanguageModelingDM_NO_LABELS
)
ICCSpokenLanguageModelingDM_NO_LABELS_28_TRAIN_14_TEST.update(
    deepcopy(ICCSpokenLanguageModelingDM_NO_LABELS_28_TRAIN_14_TEST)
)


##################
# MODELS
##################


# This version is the same as the transformers version.
GPT2_CONFIG = {
    "load": {
        "model_checkpoint": "gpt2",
        "tokenizer_checkpoint": "gpt2",
        "tokenizer_pad_token": None,
        "tokenizer_eos_token": None,
        "tokenizer_additional_special_tokens": [],
    },
    "finetune": {
        # "dm": MISSING,
        # "save_dir": MISSING,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 300,
        "run_name": None,
    },
}


GPT2_CONFIG_LABELS = deepcopy(GPT2_CONFIG)
GPT2_CONFIG_LABELS["load"] = (
    {
        "model_checkpoint": "gpt2",
        "tokenizer_checkpoint": "gpt2",
        "tokenizer_pad_token": "<PAD>",
        "tokenizer_eos_token": "<|endoftext|>",
        "tokenizer_additional_special_tokens": ["<SP1>", "<SP2>", "<START>", "<END>"],
    },
)


TURNGPT_LMMODEL_CONFIG = {
    "load": {
        "pretrained_model_name_or_path": "gpt2",
        "model_head": "LMHead",
        "load_pretrained_configs": True,
        "omit_speaker_ids": False,
        "no_train_first_n": 5,
        "learning_rate": 1e-4,
        "tokenizer_additional_special_tokens": [],
    },
    "finetune": {
        # "dm": MISSING,
        # "save_dir": MISSING,
        "max_epochs": 1,
        "log_every_n_steps": 1,
        "report_to": "wandb",
    },
}


##################
# PIPELINES
##################

CPPipeline_CONFIGS = {
    "N": -1,
    "context_buffer_size": 512,
}


##################
# EXPERIMENTS
##################
