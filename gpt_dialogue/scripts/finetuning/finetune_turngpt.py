# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-08 11:58:20
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-11 11:21:44


#############################################################
# Finetuning script for TurnGPT variants that uses TurnGPTFinetuning Data Module
# to train and store the model.
#############################################################

from omegaconf import DictConfig
import hydra
import os

# Pytorch imports
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from gpt_dialogue.turngpt.model import (
    TurnGPTDoubleHeadsModel,
    TurnGPTLMHeadModel,
)
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM

import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

############################# GLOBAL VARS. ##############################

HYDRA_CONFIG_RELATIVE_DIR = "../../conf"
HYDRA_CONFIG_NAME = "finetune_turngpt"

# NOTE: For GPU Support, this script disables tokenizer parallelism by default.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

############################# MAIN METHODS ##############################


# Load the configurations and start finetuning.
@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def turngpt_finetune(cfg : DictConfig):

    # Load the appropriate model
    model_head = cfg.finetune.model.model_head
    cfg.finetune.model.model_head.pop('model_head', None)
    if model_head == "DoubleHeads":
        model = TurnGPTDoubleHeadsModel(**cfg.finetune.model)
    elif model_head == "LMHead":
        model = TurnGPTLMHeadModel(**cfg.finetune.model)
    else:
        raise NotImplementedError(
            f"TurnGPT with head {cfg.finetune.model_head} has not been implemented"
        )
    logger.info(
        f"Loaded TurnGPT from pretrained: {cfg.finetune.model.pretrained_model_name_or_path} "
        f"with head {cfg.finetune.model_head}"
    )

    # Load the data module
    logger.info(f"Loading finetuning data module...")
    dm = TurnGPTFinetuneDM(
        tokenizer=model._tokenizer,
        train_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.train_path),
        val_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.validation_path),
        save_dir=os.getcwd(),
        **cfg.finetune.dm
    )

    logger.info("Initializing trainer...")
    # Create the loggers
    training_logger = CSVLogger(save_dir=os.getcwd(),
        name=f"TurnGPT_{cfg.finetune.model.pretrained_model_name_or_path}_{cfg.finetune.model_head}")
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=training_logger,
        log_every_n_steps=1,
        **cfg.finetune.training
    )

    logger.info("Starting training...")
    trainer.fit( model, datamodule=dm)
    logger.info("Completed!")

if __name__ == "__main__":
    turngpt_finetune()