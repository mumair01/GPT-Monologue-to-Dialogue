# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 17:27:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 14:37:44
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Callable

# TODO: Need this to access the scripts.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from LLMDialogue.pipelines import CPPipelineConfig, CPPipeline
from LLMDialogue.models import GPT2, TurnGPT, LanguageModel
from LLMDialogue.dm.ICCdm import ICCHFLMDM, ICCSpokenLanguageModelingDM
from LLMDialogue.datasets.speakerIdentity import SpeakerIdentityDataset, objectify
from LLMDialogue.datasets import load
from LLMDialogue.pipelines import (
    LLMFinetunePipeline,
    LLMFinetunePipelineConfig,
)
from LLMDialogue.pipelines.utils import run_pipeline_with_tracker
from LLMDialogue.utils.logging import (
    configure_logger,
    create_file_handler,
    create_stream_handler,
)
from LLMDialogue.tasks import CPInference_task

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


CP_INFERENCE_CONFIGS = {
    "seed": GLOBAL_SEED,
    "save_dir": "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/CPInference",
}

CP_INFERENCE_CONFIGS_NO_LABELS = deepcopy(CP_INFERENCE_CONFIGS)
CP_INFERENCE_CONFIGS_NO_LABELS["dataset"] = (
    {"name": "speakerIdentity", "configs": SD_DATASET_NO_LABELS},
)
CP_INFERENCE_CONFIGS_LABELS = deepcopy(CP_INFERENCE_CONFIGS)
CP_INFERENCE_CONFIGS_LABELS["dataset"] = {
    "name": "speakerIdentity",
    "configs": SD_DATASET_LABELS,
}


CP_INFERENCE_CONFIGS_LABELS_GPT2 = deepcopy(CP_INFERENCE_CONFIGS_LABELS)
CP_INFERENCE_CONFIGS_LABELS_GPT2["model"] = {
    "name": "gpt2",
    "configs": GPT2_CONFIG_LABELS,
}

CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT = deepcopy(CP_INFERENCE_CONFIGS_LABELS)
CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT["model"] = {
    "name": "turngpt",
    "configs": TURNGPT_LMMODEL_CONFIG,
}

####
# Create configs for each individual model - there should be 4 total.
####


## GPT-2
CP_INFERENCE_CONFIGS_LABELS_GPT2_28_TRAIN_14_TEST = deepcopy(
    CP_INFERENCE_CONFIGS_LABELS_GPT2
)
CP_INFERENCE_CONFIGS_LABELS_GPT2_28_TRAIN_14_TEST["model"]["configs"]["load"][
    "model_checkpoint"
] = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/64dec4d31ab512df25c35f96afedd432193fd81f03a2cd029ff5ad99aeaceb15/24_05_2024__15_24_55/finetune"

CP_INFERENCE_CONFIGS_LABELS_GPT2_5_TRAIN_37_TEST = deepcopy(
    CP_INFERENCE_CONFIGS_LABELS_GPT2
)
CP_INFERENCE_CONFIGS_LABELS_GPT2_5_TRAIN_37_TEST["model"]["configs"]["load"][
    "model_checkpoint"
] = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/a4ba4eb907468e565b3ca9cd0a5bd169449c7c62acdfce682830be395aca6438/24_05_2024__15_59_35/finetune"


## TURN GPT
CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_28_TRAIN_14_TEST = deepcopy(
    CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT
)
CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_28_TRAIN_14_TEST["model"]["configs"]["load"][
    "pretrained_model_name_or_path"
] = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/4faf8fdfa7450c7bc6fa6fc24bb5ad79bde3f2ceb7e87ca46a898dd395a2269d/24_05_2024__13_35_07/finetune/lightning_logs/j8szn86m/checkpoints/last.ckpt"

CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_5_TRAIN_37_TEST = deepcopy(
    CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT
)
CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_5_TRAIN_37_TEST["model"]["configs"]["load"][
    "pretrained_model_name_or_path"
] = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/ef4c0dbb3c0be6175ad7a0e36ba1a9181aeba674df2c27678a14810ccc704683/24_05_2024__14_10_25/finetune/lightning_logs/5wsg8uy1/checkpoints/last.ckpt"


_MODELS = {
    "gpt2": GPT2,
    "turngpt": TurnGPT,
}


##########################
# SPECIFIC HELPER FUNCTIONS
##########################


@hydra.main(
    config_path="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/configs/hydra",
    version_base=None,
    config_name="config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    CPInference_task(cfg)


if __name__ == "__main__":
    main()

    # jobs = [
    #     CP_INFERENCE_CONFIGS_LABELS_GPT2_5_TRAIN_37_TEST,
    #     CP_INFERENCE_CONFIGS_LABELS_GPT2_28_TRAIN_14_TEST,
    #     CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_5_TRAIN_37_TEST,
    #     CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_28_TRAIN_14_TEST,
    # ]
    # for job in jobs:
    #     conf = OmegaConf.create(job)
    #     CPInference(conf)
