# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-16 12:54:06


import sys
import os
from typing import List, Callable

import numpy as np
import pandas as pd
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from functools import partial
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline

from scripts.utils.decorators import log_wandb

############################### LOGGING SETUP #############################

logger = logging.getLogger(__name__)



########################## GLOBAL VARS. ####################################

HYDRA_CONFIG_RELATIVE_DIR = "../conf"
HYDRA_CONFIG_NAME = "config"


WANDB_PROJECT = "GPT-Monologue-Dialogue-Inference"
WANDB_ENTITY = "gpt-monologue-dialogue"


########################### HELPER METHODS ####################################

# NOTE: Assuming that the dataset is in the correct format
# and contains the columns: ["convName","convID", "Utterance"]
# TODO: Standardize the loader functions for the datasets.
def load_inference_dataset(
        csv_path : str,
        start_conv_no : int = 0,
        end_conv_no : int = -1,
        conv_key : str = "convID"
    ) -> List[pd.DataFrame]:
    """
    Load the inference dataset from a single CSV - assuming that the csv has
    separated the conversations by a unique key and contains all the utterances
    for the conversations.

    Args:
        csv_path (str)
        start_conv_no (int): Number of the conversation to start from
            Set to 0 for the first conversation.
        end_conversation_no (int): Number of the final conversation to process.
            Set to -1 to process all conversations.
        conv_key (str): Key for the conversation unique id in the dataset.
    """
    df = pd.read_csv(csv_path,index_col=0)
    conversation_dfs = [df.loc[df[conv_key] == i] for i in range(
        np.max(df[conv_key].unique()) + 1)]
    if end_conv_no > len(conversation_dfs) or end_conv_no == -1:
        end_conv_no = len(conversation_dfs)
    assert len(conversation_dfs) >= end_conv_no
    assert start_conv_no < end_conv_no
    conversation_dfs = conversation_dfs[start_conv_no:end_conv_no]
    return conversation_dfs

def generate_probabilities(
        conversation_dfs : List[pd.DataFrame],
        pipe : ConditionalProbabilityPipeline,
        save_dir : str,
        run : wandb.run
    ) -> None:
    """
    Use the conditional probability pipeline to generate probabilities for all
    conversations individually i.e., the conversations are treated as independent.
    """
    data = []
    for i, conversation_df in enumerate(conversation_dfs):
        utterances = list(conversation_df["Utterance"])
        res = pipe(
            utterances=utterances
        )
        save_path = os.path.join(save_dir,
                "{}_conditional_probs.csv".format(conversation_df["convName"].iloc[0]))
        # columns = ["conversation_name"] + list(res[0].keys())
        df = pd.DataFrame(data=res, columns=list(res.raw_output()[0].keys()))
        df["conversation_name"] = conversation_df["convName"].iloc[0]
        df["conversation_number"] = i
        # Set the order of the columns
        df = df[[
            'conversation_number','conversation_name', 'turn_no','word_no',
            'context','word','last_word_prob' ]]
        df.to_csv(save_path)
        data.append(df.copy())

        run.log({
            f"overview/steps" : i,
            f"overview/turns_at_step" : len(df)
        })

    results_df = pd.concat(data)
    results_df.to_csv(
        os.path.join(save_dir,"conditional_probs_combined.csv"))

    wandb_table = wandb.Table(data=results_df)
    run.log({
        f"tables/combined_results" :  wandb_table,
    })


########################### MAIN METHODS ####################################

def generate_wandb_run_name(cfg):
    return f"{cfg.experiment.name}_{cfg.experiment.model_name}_{cfg.dataset.name}"


@log_wandb(
    logger=logger,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    wandb_init_mode="disabled",
    run_name_func=generate_wandb_run_name
)
def run_inference(cfg : DictConfig, run : wandb.run):
    logger.info("Running inference with configurations:")
    print(OmegaConf.to_yaml(cfg))

    # Load the appropriate model
    if cfg.experiment.name == "inference_monologue_gpt":
        model = MonologueGPT()
    elif cfg.experiment.name == "inference_turngpt":
        model = TurnGPT()
        # NOTE: Augmenting the native tokenizer so that the correct Args
        # are used for the model tokenizer.
        # IMPORTANT: This is based on the assumption that the input
        # data contains <ts> / EOS tokens both for speaker continuations and
        # at the end of turns.
        def _turngpt_encode_wrapper(text, *args,**kwargs):
            return model.tokenizer(
                text,
                *args,
                add_prefix_space=True,
                add_eos_token=False,
                return_token_type_ids=True,
                # NOTE: This is important - for our experiments with TurnGPT,
                # we do not want to split speaker by the inline eos token.
                split_speaker_by_inline_eos=False,
                **kwargs
            )
        model.encode = _turngpt_encode_wrapper
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.name} not defined"
        )

    # Load the model
    model.load(**OmegaConf.to_object(cfg.experiment.load))
    logger.info(f"Loading model of type: {model}")

    # Load the pipeline, the dataset, and execute the task
    pipe = ConditionalProbabilityPipeline(
        model=model,
        **cfg.experiment.inference
    )
    logger.info(f"Starting inference using pipe: {pipe}")

    conversation_dfs = load_inference_dataset(
        csv_path = os.path.join(cfg.env.paths.root, cfg.dataset.dataset_path),
        **cfg.experiment.dataset
    )
    logger.info(f"Loaded {len(conversation_dfs)} datasets to generate probabilities")

    generate_probabilities(
        conversation_dfs=conversation_dfs,
        pipe=pipe,
        save_dir=os.getcwd(),
        run=run
    )
    logger.info("Completed inference!")


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def main(cfg : DictConfig):
    run_inference(cfg)

if __name__ == "__main__":
    main()


