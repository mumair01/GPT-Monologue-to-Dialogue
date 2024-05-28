# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 17:27:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-24 16:53:28
import sys
import os
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


def get_config_hash(config: DictConfig) -> str:
    """
    Generates a unique hash for a DictConfig with potentially nested dictionaries.
    """

    def hash_dict(d):
        if isinstance(d, DictConfig):
            sorted_subdict = {k: hash_dict(v) for k, v in sorted(d.items())}
            return hash(json.dumps(sorted_subdict))
        else:
            return str(d)

    sorted_dict = {k: hash_dict(v) for k, v in sorted(config.items())}
    json_str = json.dumps(sorted_dict)
    return hash(json_str)


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


##########################
# TASK FUNCTION
##########################


def CPInference(
    cfg: DictConfig,
):
    logger.info("Starting CPInference experiment with configurations:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Update the save directory path using unique arguments and date/time.
    now = datetime.now()
    save_dir = (
        f"{cfg.save_dir}/{get_config_hash(cfg)}/{now.strftime('%d_%m_%Y__%H_%M_%S')}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Load the model first.
    model_conf = OmegaConf.to_object(cfg.model.configs.load)
    model = _MODELS[cfg.model.name](cfg.seed)
    logger.info("Loading model with configurations:")
    logger.info(model_conf)
    logger.warning(
        f"Make sure that the correct model is loaded if you intended to load a finetuned model! "
    )
    model.load(**model_conf)

    # Load the dataset
    print(cfg.dataset.configs)
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
        save_dir=save_dir,
        **CPPipeline_CONFIGS,
    )

    # Save the configurations file.
    with open(f"{save_dir}/CPInference_configs.yaml", "w") as fp:
        OmegaConf.save(config=conf, f=fp.name)
        loaded = OmegaConf.load(fp.name)
        assert conf == loaded

    logger.info(f"Experiment completed - results path: {save_dir}")


if __name__ == "__main__":
    jobs = [
        CP_INFERENCE_CONFIGS_LABELS_GPT2_5_TRAIN_37_TEST,
        CP_INFERENCE_CONFIGS_LABELS_GPT2_28_TRAIN_14_TEST,
        CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_5_TRAIN_37_TEST,
        CP_INFERENCE_CONFIGS_NO_LABELS_TURNGPT_28_TRAIN_14_TEST,
    ]
    for job in jobs:
        conf = OmegaConf.create(job)
        CPInference(conf)
