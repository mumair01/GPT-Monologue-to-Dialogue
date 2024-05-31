# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-29 16:30:23
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-31 13:19:39


import sys
import os
from omegaconf import DictConfig, OmegaConf
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Callable

from LLMDialogue.baseConfigs import BaseConfigs
from LLMDialogue.pipelines import CPPipelineConfig, CPPipeline
from LLMDialogue.models import GPT2, TurnGPT, LanguageModel
from LLMDialogue.dm.ICCdm import ICCHFLMDM, ICCSpokenLanguageModelingDM
from LLMDialogue.datasets.speakerIdentity import SpeakerIdentityDataset, objectify
from LLMDialogue.datasets import load
from LLMDialogue.pipelines import (
    LLMFinetunePipeline,
    LLMFinetunePipelineConfig,
)
from LLMDialogue.pipelines.utils import run_pipeline_with_tracker, get_dirname
from LLMDialogue.utils.logging import (
    configure_logger,
    create_file_handler,
    create_stream_handler,
)
from LLMDialogue.utils.general import save_dict_conf


import logging

logger = logging.getLogger(__name__)


_MODELS = {
    "gpt2": GPT2,
    "turngpt": TurnGPT,
}


##########################
# SPECIFIC HELPER FUNCTIONS
##########################


def get_config_hash(config: DictConfig) -> str:
    """
    Generates a unique hash for a DictConfig with potentially nested dictionaries,
    ensuring all hashes are positive.
    """

    def hash_dict(d):
        if isinstance(d, DictConfig):
            sorted_subdict = {k: hash_dict(v) for k, v in sorted(d.items())}
            # Use bitwise AND with a large positive number to keep the hash positive
            return hash(json.dumps(sorted_subdict)) & 0x7FFFFFFF
        else:
            return str(d)

    sorted_dict = {k: hash_dict(v) for k, v in sorted(config.items())}
    json_str = json.dumps(sorted_dict)
    # Use bitwise AND with a large positive number to keep the hash positive
    return hash(json_str) & 0x7FFFFFFF


def predict_speaker_identity_dset(
    model: LanguageModel,
    dset: SpeakerIdentityDataset,
    save_dir: str,
    N: int,
    context_buffer_size: int,
    probability_method: str,
) -> Tuple:
    assert isinstance(model, LanguageModel)
    assert isinstance(dset, SpeakerIdentityDataset)
    splits = objectify(dset)
    test_df = splits.data_df_map["test"]
    # This is a list of all of the conversation IDs.
    conv_ids = list(set(test_df["conversationID"].tolist()))
    results = list()
    for conv_id in conv_ids:
        # Get the utterances for this conversation.
        utterances = test_df[test_df["conversationID"] == conv_id]["utterance"].tolist()
        # Set up the pipeline.
        pipe_config = CPPipelineConfig(
            model=model,
            utterances=utterances,
            N=N,
            context_buffer_size=context_buffer_size,
            probability_method=probability_method,
            # TODO: I can index this so that each individual json is saved as well.
            cache_dir=f"{save_dir}/stimuli/{conv_id}",
        )
        pipe = CPPipeline(pipe_config)
        pipe.run()
        res = pipe.info().data
        # Create a dataframe from the results and save.
        res_df = pd.DataFrame(res)
        results.append(res_df.copy())
    # Concat the df into a larger one and save
    df = pd.concat(results)
    save_fpath = f"{save_dir}/CP.csv"
    df.to_csv(save_fpath)
    return df, save_fpath


def CPInference_task(
    cfg: DictConfig,
):
    logger.info("Starting CPInference experiment with configurations:")
    logger.info(OmegaConf.to_yaml(cfg))

    root_save_dir = f"{BaseConfigs.pipeline.cachePath}Inference"
    os.makedirs(root_save_dir, exist_ok=True)

    # Load the model first.
    model_conf = OmegaConf.to_object(cfg.model.load)

    model = _MODELS[cfg.model_type](cfg.env.seed)
    logger.info("Loading model with configurations:")
    logger.info(model_conf)
    logger.warning(
        f"Make sure that the correct model is loaded if you intended to load a finetuned model! "
    )
    model.load(**model_conf)

    # Update the save directory path using unique arguments and date/time.
    now = datetime.now()
    save_dir = f"{root_save_dir}/CP_{cfg.model_type}_tok={cfg.dataset.add_eos_tok_bw_turns}_method={cfg.inference.pipeline.probability_method}/{get_config_hash(cfg)}/{now.strftime('%d_%m_%Y__%H_%M_%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Load the dataset
    OmegaConf.update(cfg, "dataset.seed", cfg.env.seed)
    dset_conf = OmegaConf.to_object(cfg.dataset)
    dset_conf["data_dir"] = os.path.join(BaseConfigs.root_path, dset_conf["data_dir"])
    logger.info("Loading the dataset with configurations:")
    logger.info(dset_conf)
    dset = load(
        dset_name=cfg.inference.dset_name,
        config_name=dset_conf["variant_name"],
        **dset_conf,
    )

    # Run the inference function.
    logger.info("Starting to run the prediction pipeline...")
    predict_speaker_identity_dset(
        model=model,
        dset=dset,
        save_dir=save_dir,
        **cfg.inference.pipeline,
    )

    # Save the configurations file.
    save_fpath = f"{save_dir}/CPInference_configs.yaml"
    save_dict_conf(save_fpath, cfg)
    logger.info(f"Saved experiment configurations to: {save_fpath}")

    logger.info(f"Experiment completed - results path: {save_dir}")
