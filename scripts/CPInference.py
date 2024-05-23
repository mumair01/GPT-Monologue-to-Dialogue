# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 17:27:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 17:50:04
import sys
import os
from omegaconf import DictConfig, OmegaConf
import pandas as pd
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
    "model": {
        "name": "gpt2",
        "configs": GPT2_CONFIG_LABELS,
    },
    "dataset": {"name": "speakerIdentity", "configs": SD_DATASET_LABELS},
}

FINETUNE_GPT2_LABELS_28_TRAIN_14_TEST = {
    "seed": GLOBAL_SEED,
    "model": {
        "name": "gpt2",
        "configs": GPT2_CONFIG_LABELS,
    },
    "dataset": {"name": "speakerIdentity", "configs": SD_DATASET_LABELS},
}

FINETUNE_TURNGPT_NO_LABELS_5_TRAIN_37_TEST = {
    "seed": GLOBAL_SEED,
    "model": {
        "name": "turngpt",
        "configs": TURNGPT_LMMODEL_CONFIG,
    },
    "dataset": {"name": "speakerIdentity", "configs": SD_DATASET_NO_LABELS},
}


FINETUNE_TURNGPT_NO_LABELS_28_TRAIN_14_TEST = {
    "seed": GLOBAL_SEED,
    "model": {
        "name": "turngpt",
        "configs": TURNGPT_LMMODEL_CONFIG,
    },
    "dataset": {"name": "speakerIdentity", "configs": SD_DATASET_NO_LABELS},
}


_MODELS = {
    "gpt2": GPT2,
    "turngpt": TurnGPT,
}

##########################
# TASK FUNCTION
##########################

# TODO: Check to see how I can create this directory dynamically.
DIR = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/CPInference"
os.makedirs(DIR, exist_ok=True)


def predict_speaker_identity_dset(
    model: LanguageModel,
    dset: SpeakerIdentityDataset,
    save_dir: str,
    N: int = -1,
    context_buffer_size: int = 512,
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
            # TODO: I can index this so that each individual json is saved as well.
            cache_dir=f"{DIR}/{conv_id}",
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


def CPInference(cfg: DictConfig):
    logger.info("Starting CPInference experiment with configurations:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Load the model first.
    model_conf = OmegaConf.to_object(cfg.model.configs.load[0])
    model = _MODELS[cfg.model.name](cfg.seed)
    logger.info("Loading model with configurations:")
    logger.info(model_conf)
    logger.warning(
        f"Make sure that the correct model is loaded if you intended to load a finetuned model! "
    )
    model.load(**model_conf)

    # Load the dataset
    dset_conf = OmegaConf.to_object(cfg.dataset.configs)
    dset = load(
        dset_name=cfg.dataset.name,
        config_name=dset_conf["variant_name"],
        **dset_conf,
    )

    # Run the inference function.
    predict_speaker_identity_dset(
        model=model,
        dset=dset,
        save_dir=DIR,
        **CPPipeline_CONFIGS,
    )

    # Save the configurations file.
    with open(f"{DIR}/CPInference_configs.yaml", "w") as fp:
        OmegaConf.save(config=conf, f=fp.name)
        loaded = OmegaConf.load(fp.name)
        assert conf == loaded

    logger.info(f"Experiment completed - results path: {DIR}")


if __name__ == "__main__":
    conf = OmegaConf.create(FINETUNE_GPT2_LABELS_28_TRAIN_14_TEST)
    CPInference(conf)
