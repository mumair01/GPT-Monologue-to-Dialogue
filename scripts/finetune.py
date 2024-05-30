# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 15:33:54
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 13:29:46


import sys
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Callable

# TODO: Need this to access the scripts.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from LLMDialogue.tasks import finetune_task
from LLMDialogue.utils.logging import (
    configure_logger,
    create_file_handler,
    create_stream_handler,
)
from scripts.configs import *

import logging

logger = logging.getLogger(__name__)


configure_logger(
    handlers=[
        create_stream_handler(logging_level=logging.INFO),
        # create_file_handler(
        #     outdir=LOGGING_DIR,
        #     logging_level=logging.DEBUG,
        #     reset_outdir=False,  # Keep logs across runs.
        # ),
    ]
)

##########################
# CONFIGURATION DICTIONARY
##########################

SAVE_DIR = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/outputs"

FINETUNE_GPT2_LABELS_5_TRAIN_37_TEST = {
    "seed": GLOBAL_SEED,
    "tracker": {
        "tracker_type": "wandb",
        "configs": deepcopy(WANDB_CONFIGS),
    },
    "model": {
        "name": "gpt2",
        "configs": GPT2_CONFIG_LABELS,
    },
    "dm": {
        "name": "ICCHFLMDM",
        "configs": ICCHFLMDM_LABELS_5_TRAIN_37_TEST,
    },
}

FINETUNE_GPT2_LABELS_28_TRAIN_14_TEST = {
    "seed": GLOBAL_SEED,
    "tracker": {
        "tracker_type": "none",
        "configs": deepcopy(WANDB_CONFIGS),
    },
    "model": {
        "name": "gpt2",
        "configs": GPT2_CONFIG_LABELS,
    },
    "dm": {
        "name": "ICCHFLMDM",
        "configs": ICCHFLMDM_LABELS_28_TRAIN_14_TEST,
    },
}

FINETUNE_TURNGPT_NO_LABELS_5_TRAIN_37_TEST = {
    "seed": GLOBAL_SEED,
    "tracker": {
        "tracker_type": "wandb",
        "configs": deepcopy(WANDB_CONFIGS),
    },
    "model": {
        "name": "turngpt",
        "configs": TURNGPT_LMMODEL_CONFIG,
    },
    "dm": {
        "name": "ICCSpokenLanguageModelingDM",
        "configs": ICCSpokenLanguageModelingDM_NO_LABELS_5_TRAIN_37_TEST,
    },
}


FINETUNE_TURNGPT_NO_LABELS_28_TRAIN_14_TEST = {
    "seed": GLOBAL_SEED,
    "tracker": {
        "tracker_type": "wandb",
        "configs": deepcopy(WANDB_CONFIGS),
    },
    "model": {
        "name": "turngpt",
        "configs": TURNGPT_LMMODEL_CONFIG,
    },
    "dm": {
        "name": "ICCSpokenLanguageModelingDM",
        "configs": ICCSpokenLanguageModelingDM_NO_LABELS_28_TRAIN_14_TEST,
    },
}


##########################
# TASK FUNCTION
##########################


@hydra.main(
    config_path="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/configs/hydra",
    version_base=None,
    config_name="config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune_task(cfg)


if __name__ == "__main__":
    main()

    # # Finetune using all of the configs
    # jobs = [
    #     FINETUNE_GPT2_LABELS_28_TRAIN_14_TEST,
    #     # FINETUNE_GPT2_LABELS_5_TRAIN_37_TEST,
    #     # FINETUNE_TURNGPT_NO_LABELS_28_TRAIN_14_TEST,
    #     # FINETUNE_TURNGPT_NO_LABELS_5_TRAIN_37_TEST,
    # ]

    # for job in jobs:
    #     conf = OmegaConf.create(job)
    #     finetune_task(conf)
