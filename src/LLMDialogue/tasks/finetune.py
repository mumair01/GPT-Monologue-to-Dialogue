# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-29 16:30:13
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 14:17:29


import sys
import os
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Callable
from copy import deepcopy


from LLMDialogue.pipelines import CPPipelineConfig, CPPipeline
from LLMDialogue.models import GPT2, TurnGPT, LanguageModel
from LLMDialogue.dm.ICCdm import ICCHFLMDM, ICCSpokenLanguageModelingDM
from LLMDialogue.datasets.speakerIdentity import SpeakerIdentityDataset, objectify
from LLMDialogue.pipelines import (
    LLMFinetunePipeline,
    LLMFinetunePipelineConfig,
)
from LLMDialogue.baseConfigs import BaseConfigs
from LLMDialogue.pipelines.utils import run_pipeline_with_tracker
from LLMDialogue.utils.logging import (
    configure_logger,
    create_file_handler,
    create_stream_handler,
)

from LLMDialogue.utils.general import remove_keys_from_dict, save_dict_conf


import logging

logger = logging.getLogger(__name__)


_MODELS = {
    "gpt2": GPT2,
    "turngpt": TurnGPT,
}

_DMS = {
    "ICCHFLMDM": ICCHFLMDM,
    "ICCSLFTDM": ICCSpokenLanguageModelingDM,
}


# This needs to be set to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def finetune_task(cfg: DictConfig) -> None:
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

    ######## Load the model first.

    model_conf = OmegaConf.to_object(cfg.model.load)
    model = _MODELS[cfg.model_type](cfg.env.seed)
    logger.info("Loading model with configurations:")
    logger.info(model_conf)
    model.load(**model_conf)

    ######## Load the data module.
    dm_configs = OmegaConf.to_object(cfg.dm)
    dataset_configs = OmegaConf.to_object(cfg.dataset)
    dataset_configs["data_files"] = {
        k: os.path.join(BaseConfigs.root_path, v)
        for k, v in dataset_configs["data_files"].items()
    }
    dataset_configs["seed"] = cfg.env.seed
    OmegaConf.update(cfg, "dataset", dataset_configs)
    dm_configs.update(
        {
            "dset_variant": cfg.dataset.variant_name,
            "tokenizer": model.tokenizer,
            "save_dir": f"{BaseConfigs.datasets.cachePath}/dm",
        }
    )
    dm_configs_copy = deepcopy(dm_configs)
    dm_configs_copy.pop("tokenizer")
    dm_configs.update(**dataset_configs)
    OmegaConf.update(cfg, "dm", dm_configs_copy)

    logger.info(f"Loading data module of type {_DMS[cfg.dm_type]} with configurations:")
    logger.info(dm_configs)
    dm = _DMS[cfg.dm_type](**dm_configs)

    # Create the finetune pipeline
    logger.info("Loading LLM Fine-tuning pipeline")
    pipe_config = LLMFinetunePipelineConfig(
        model=model,
        dm=dm,
        # NOTE: If we want to disable the tracker, then it needs to be done here as well.
        report_to=cfg.tracker_type,
        finetune_kwargs=OmegaConf.to_object(cfg.model.finetune),
    )
    finetune_pipeline = LLMFinetunePipeline(pipe_config)

    logger.info(f"Using MLTracker of type: {cfg.tracker_type}")
    tracker_kwargs = OmegaConf.to_object(cfg.tracker)
    logging_dir = f"{pipe_config.cache_dir}/trainer_logs"
    os.makedirs(logging_dir, exist_ok=True)
    tracker_kwargs.update(
        {
            "logging_dir": logging_dir,
            # Give a nice, readable run name that identifies the experiment.
            "run_name": f"{cfg.model_type}_{cfg.dm_type}_{dm_configs['dset_variant']}",
        }
    )
    OmegaConf.update(cfg, "tracker", tracker_kwargs)
    tracker_kwargs["configs"] = cfg
    logger.info(tracker_kwargs)
    run_pipeline_with_tracker(
        pipe=finetune_pipeline,
        tracker_type=cfg.tracker_type,
        **tracker_kwargs,
    )

    # Save the configurations file.
    save_dict_conf(f"{pipe_config.cache_dir}/finetune_configs.yaml", cfg)

    logger.info(f"Experiment completed - results path: {pipe_config.cache_dir}")
