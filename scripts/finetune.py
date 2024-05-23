# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 15:33:54
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 18:55:25


import sys
import os
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Callable

# TODO: Need this to access the scripts.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from LLMDialogue.pipelines import CPPipelineConfig, CPPipeline
from LLMDialogue.models import GPT2, TurnGPT, LanguageModel
from LLMDialogue.dm.ICCdm import ICCHFLMDM, ICCSpokenLanguageModelingDM
from LLMDialogue.datasets.speakerIdentity import SpeakerIdentityDataset, objectify
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
    "tracker": {"tracker_type": "wandb", "configs": deepcopy(WANDB_CONFIGS)},
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
        "tracker_type": "wandb",
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


_MODELS = {
    "gpt2": GPT2,
    "turngpt": TurnGPT,
}

_DMS = {
    "ICCHFLMDM": ICCHFLMDM,
    "ICCSpokenLanguageModelingDM": ICCSpokenLanguageModelingDM,
}


def finetune(cfg: DictConfig):
    """
    This is the task function for fine-tuning.
    TODO: Port this to work with hydra zen later.

    Parameters
    ----------
    cfg : DictConfig
        A configuration object that defines each component for the experiment.
    """
    logger.info("Starting finetuning experiment with configurations:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Load the model first.

    model_conf = OmegaConf.to_object(cfg.model.configs.load)
    model = _MODELS[cfg.model.name](cfg.seed)
    logger.info("Loading model with configurations:")
    logger.info(model_conf)
    model.load(**model_conf)

    # Load the data module.
    dm_configs = OmegaConf.to_object(cfg.dm.configs)
    dm_configs.update(
        {
            "tokenizer": model.tokenizer,
            "save_dir": f"{BaseConfigs.datasets.cachePath}/dm",
        }
    )
    logger.info(
        f"Loading data module of type {type(_DMS[cfg.dm.name])} with configurations:"
    )
    logger.info(dm_configs)
    dm = _DMS[cfg.dm.name](**dm_configs)
    # Create the finetune pipeline
    logger.info("Loading LLM Fine-tuning pipeline")
    pipe_config = LLMFinetunePipelineConfig(
        model=model,
        dm=dm,
        report_to=cfg.tracker.tracker_type,
        finetune_kwargs=OmegaConf.to_object(cfg.model.configs.finetune),
    )
    finetune_pipeline = LLMFinetunePipeline(pipe_config)

    logger.info(f"Using MLTracker of type: {cfg.tracker.tracker_type}")
    tracker_kwargs = OmegaConf.to_object(cfg.tracker.configs)
    tracker_kwargs.update({"logging_dir": pipe_config.cache_dir})
    run_pipeline_with_tracker(
        tracker_type=cfg.tracker.tracker_type,
        pipe=finetune_pipeline,
        **tracker_kwargs,
    )

    # Save the configurations file.
    with open(f"{pipe_config.cache_dir}/finetune_configs.yaml", "w") as fp:
        OmegaConf.save(config=conf, f=fp.name)
        loaded = OmegaConf.load(fp.name)
        assert conf == loaded

    logger.info(f"Experiment completed - results path: {pipe_config.cache_dir}")


if __name__ == "__main__":

    # Finetune using all of the configs
    jobs = [
        # FINETUNE_GPT2_LABELS_28_TRAIN_14_TEST,
        # FINETUNE_GPT2_LABELS_5_TRAIN_37_TEST,
        FINETUNE_TURNGPT_NO_LABELS_28_TRAIN_14_TEST,
        # FINETUNE_TURNGPT_NO_LABELS_5_TRAIN_37_TEST,
    ]

    for job in jobs:
        conf = OmegaConf.create(job)
        finetune(conf)
