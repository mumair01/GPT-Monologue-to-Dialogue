# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-11-30 08:45:37
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-16 14:15:02

import sys
import os

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import logging

from typing import Union, Callable


def log_wandb(
    logger: logging.Logger,
    wandb_project: str,
    wandb_entity: str,
    wandb_init_mode: Union[str, None],
    run_name_func: Callable,
):
    def func_wrapper(func):
        def func_args_wrapper(cfg: DictConfig, *args, **kwargs):
            logger.info("NOTE: Logging using Weights and Biases (WANDB)")

            # Log the config params using wandb
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )

            # Initialize the wandb instance given the mode.
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                ),
                mode=wandb_init_mode,
            )

            # Change the run name
            run.name = f"{run_name_func(cfg, *args, **kwargs)}_{run.id}"
            run.save()

            logger.info(
                f"WANDB: Running experiment:\n"
                f"\tProject: {wandb_project}\n"
                f"\tEntity: {wandb_entity}\n"
                f"\tID: {run.id}\n"
                f"\tName: {run.name}"
            )

            func(cfg, run)

            # Finish logging the run
            logger.info(f"WANDB: Ending logging for experiment: {run.id}")
            run.finish()

        return func_args_wrapper

    return func_wrapper
