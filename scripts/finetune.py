# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 12:19:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-17 08:46:15

""" 
Assumptions
-----------
1. Before this script is run, it assumed that the environment has been 
    set using set_hpc_env.sh or set_env.sh
"""

import sys
import os
from functools import partial
from typing import Callable

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb
import logging

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
)

from gpt_dialogue.gpt2 import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT

from scripts.decorators import log_wandb

logger = logging.getLogger(__name__)

########################## GLOBAL VARS. ####################################

# Load Hydra and wandb args from the environment
HYDRA_CONFIG_RELATIVE_DIR = os.getenv("HYDRA_CONFIG_RELATIVE_DIR")
HYDRA_CONFIG_NAME = os.getenv("HYDRA_CONFIG_NAME")

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_INIT_MODE = os.getenv("WANDB_INIT_MODE")


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
    wandb_init_mode=WANDB_INIT_MODE,
    run_name_func=generate_wandb_run_name,
)
def run_finetuning(cfg: DictConfig, run: wandb.run):
    logger.info("Starting finetuning experiment with configurations:")
    print(OmegaConf.to_yaml(cfg))

    # Load the appropriate model
    if cfg.experiment.name == "finetune_gpt2":
        model = MonologueGPT()
    elif cfg.experiment.name == "finetune_turngpt":
        model = TurnGPT()

        # NOTE: Augmenting the native tokenizer so that the correct Args
        # are used for the model tokenizer.
        # IMPORTANT: This is based off the assumption that the training data
        # has <ts> / EOS tokens ONLY at turn continuations, NOT at turn ends.
        def _turngpt_encode_wrapper(**kwargs):
            return partial(
                model.tokenizer(
                    add_prefix_space=True,
                    add_eos_token=True,
                    return_token_type_ids=True,
                    # NOTE: This is important - for our experiments with TurnGPT,
                    # we do not want to split speaker by the inline eos token.
                    split_speaker_by_inline_eos=False,
                    **kwargs,
                )
            )

        model.encode = _turngpt_encode_wrapper
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.name} not defined"
        )

    # Load the model
    model.load(**OmegaConf.to_object(cfg.experiment.load))
    logger.info(f"Loading model of type: {model}")
    # Finetune
    logger.info("Starting finetuning...")
    model.finetune(
        train_csv_path=os.path.join(
            cfg.env.paths.root, cfg.dataset.train_csv_path
        ),
        val_csv_path=os.path.join(
            cfg.env.paths.root, cfg.dataset.validation_csv_path
        ),
        save_dir=os.getcwd(),
        **OmegaConf.to_object(cfg.experiment.finetune),
    )
    logger.info("Finetuning completed!")


@hydra.main(
    version_base=None,
    config_path=HYDRA_CONFIG_RELATIVE_DIR,
    config_name=HYDRA_CONFIG_NAME,
)
def main(cfg: DictConfig):
    run_finetuning(cfg)


if __name__ == "__main__":
    main()
