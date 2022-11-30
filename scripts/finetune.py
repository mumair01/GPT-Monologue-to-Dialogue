# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-11-30 09:41:02

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

from scripts.utils.decorators import log_wandb

logger = logging.getLogger(__name__)

########################## GLOBAL VARS. ####################################

HYDRA_CONFIG_RELATIVE_DIR = "../conf"
HYDRA_CONFIG_NAME = "config"

WANDB_PROJECT = "GPT-Monologue-Dialogue-Finetune"
WANDB_ENTITY = "gpt-monologue-dialogue"


########################### MAIN METHODS ####################################

def generate_wandb_run_name(cfg):
    try:
        return f"{cfg.experiment.name}_{cfg.experiment.load.model_head}_{cfg.dataset.name}"
    except:
        return f"{cfg.experiment.name}_{cfg.dataset.name}"


@log_wandb(
    logger=logger,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    wandb_init_mode=None,
    run_name_func=generate_wandb_run_name
)
def run_finetuning(cfg : DictConfig, run : wandb.run):

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