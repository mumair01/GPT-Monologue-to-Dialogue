# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-21 18:37:18

import sys
import os
from typing import Callable

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT

logger = logging.getLogger(__name__)

########################## GLOBAL VARS. ####################################

HYDRA_CONFIG_RELATIVE_DIR = "../conf"
HYDRA_CONFIG_NAME = "config"

WANDB_PROJECT = "GPT-Monologue-Dialogue"
WANDB_ENTITY = "gpt-monologue-dialogue"


########################### MAIN METHODS ####################################


def log_wandb(func : Callable):
    """Decorator for setting up and logging experiment using wandb"""

    logger.info("WANDB: Logging finetuning using Weights and Biases (WANDB)")

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
        wandb.run.name = f"{cfg.experiment.name}_{cfg.dataset.name}_{run_id}"

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

@log_wandb
def run_finetuning(cfg : DictConfig):
    logger.info("Starting finetuning experiment with configurations:")
    print(OmegaConf.to_yaml(cfg))
    # Load the appropriate model
    if cfg.experiment.name == "finetune_monologue_gpt":
        model = MonologueGPT()
    elif cfg.experiment.name == "finetune_turngpt":
        model = TurnGPT()
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.name} not defined"
        )

    # Load the model
    logger.info(f"Loading model of type: {model}")
    model.load(**cfg.experiment.load)

    # Finetune
    model.finetune(
        train_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.train_csv_path),
        val_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.validation_csv_path),
        save_dir = os.getcwd(),
        **cfg.experiment.finetune
    )
    logger.info("Finetuning completed!")


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def main(cfg : DictConfig):
    run_finetuning(cfg)

if __name__ == "__main__":
    main()