# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-22 09:42:29


import sys
import os
from typing import List, Callable

import numpy as np
import pandas as pd
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
import logging


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline

logger = logging.getLogger(__name__)

########################## GLOBAL VARS. ####################################

HYDRA_CONFIG_RELATIVE_DIR = "../conf"
HYDRA_CONFIG_NAME = "config"


WANDB_PROJECT = "GPT-Monologue-Dialogue-Inference"
WANDB_ENTITY = "gpt-monologue-dialogue"



########################### HELPER METHODS ####################################

def log_wandb(func : Callable):
    """Decorator for setting up and logging experiment using wandb"""

    logger.info("WANDB: Logging inference using Weights and Biases (WANDB)")

    def inner(cfg : DictConfig):
        # Log the config params using wandb
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ))
        # Change the run name
        run_id = wandb.run.id
        wandb.run.name = f"{cfg.experiment.name}_{cfg.experiment.model_name}_{cfg.dataset.name}_{run_id}"

        logger.info(
            f"WANDB: Running experiment for project {WANDB_PROJECT} entity "
            f"{WANDB_ENTITY} with id {wandb.run.id}"
        )

        # Run experiment
        func(cfg)

        # Finish logging the run
        logger.info(f"WANDB: Ending logging for experiment: {run_id}")
        run.finish()

    return inner


# NOTE: Assuming that the dataset is in the correct format.
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
        df = pd.DataFrame(data=res, columns=list(res[0].keys()))
        df["conversation_name"] = conversation_df["convName"].iloc[0]
        df["conversation_number"] = i
        # Set the order of the columns
        df = df[[
            'conversation_number','conversation_name', 'turn_no','word_no',
            'context','word','last_word_prob' ]]
        df.to_csv(save_path)
        data.append(df.copy())

        wandb_table = wandb.Table(data=df.copy())
        wandb.run.log({
            conversation_df["convName"].iloc[0] : wandb_table
        })


    results_df = pd.concat(data)
    results_df.to_csv(
        os.path.join(save_dir,"conditional_probs_combined.csv"))


########################### MAIN METHODS ####################################


@log_wandb
def run_inference(cfg : DictConfig):
    logger.info("Running inference with configurations:")
    print(OmegaConf.to_yaml(cfg))
    # Load the appropriate model
    if cfg.experiment.name == "inference_monologue_gpt":
        model = MonologueGPT()
    elif cfg.experiment.name == "inference_turngpt":
        model = TurnGPT()
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.name} not defined"
        )

    # Load the model
    logger.info(f"Loading model of type: {model}")
    model.load(**cfg.experiment.load)

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
        save_dir=os.getcwd()
    )
    logger.info("Completed inference!")

@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def main(cfg : DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
