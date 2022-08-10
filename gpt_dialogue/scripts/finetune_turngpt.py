# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-08 11:58:20
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-10 16:28:49


#############################################################
'''
This is a finetuning script for TurnGPT.

'''
#############################################################
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Callable
import os

# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from datasets import concatenate_datasets

from gpt_dialogue.turngpt.model import TurnGPTDoubleHeadsModel, TurnGPTLMHeadModel, TurnGPTModel
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM


HYDRA_CONFIG_RELATIVE_DIR = "../../conf"
HYDRA_CONFIG_NAME = "finetune_turngpt"

import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# NOTE: For GPU Support, this script disables tokenizer parallelism by default.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


############################# MAIN METHODS ##############################


# Load the configurations and start finetuning.
@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def turngpt_finetune(cfg : DictConfig):

    logger.info(f"Loading TurnGPT from pretrained: {cfg.finetune.model.pretrained_model_name_or_path}")
    # model = TurnGPTDoubleHeadsModel(**cfg.finetune.model)
    model = TurnGPTLMHeadModel()
    # Load the data module.
    logger.info(f"Loading finetuning data module...")
    dm = TurnGPTFinetuneDM(
        tokenizer=model._tokenizer,
        train_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.train_path),
        val_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.validation_path),
        conversation_id_key="convID",
        utterance_key="Utterance",
        use_cache=True,
        save_dir=os.getcwd(),
        **cfg.finetune.dm
    )
    logger.info("Initializing trainer...")

    # Create the loggers
    training_logger = CSVLogger(save_dir=os.getcwd(),name="test")
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=10,
        log_every_n_steps=1,
        logger=training_logger,
        **cfg.finetune.training
    )
    logging.info("Starting training...")


    trainer.fit(
        model,
        datamodule=dm
    )

if __name__ == "__main__":
    turngpt_finetune()