# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-08 11:58:20
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-08 15:03:45


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

from gpt_dialogue.turngpt.model import TurnGPT
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM


HYDRA_CONFIG_RELATIVE_DIR = "../../conf"
HYDRA_CONFIG_NAME = "finetune_turngpt"

import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


############################# MAIN METHODS ##############################

def clean_speaker_labels(data):
    """Remove speaker labels from the start and end of an utterance"""
    if len(data['Utterance'].split()) > 1:
        data['Utterance'] = " ".join(data['Utterance'].split()[1:-1])
    return data

# Load the configurations and start finetuning.
@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_DIR, config_name=HYDRA_CONFIG_NAME)
def turngpt_finetune(cfg : DictConfig):

    logger.info(f"Loading TurnGPT from pretrained: {cfg.finetune.model.pretrained_model_name_or_path}")
    print(type(cfg.finetune.model.tokenizer_additional_special_tokens))
    model = TurnGPT(**cfg.finetune.model)
    # Load the data module.
    logger.info(f"Loading finetuning data module...")
    dm = TurnGPTFinetuneDM(
        tokenizer=model._tokenizer,
        train_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.train_path),
        val_csv_path=os.path.join(cfg.env.paths.root,cfg.dataset.validation_path),
        save_dir=os.getcwd(),
        cleanup_fn=clean_speaker_labels,
        **cfg.finetune.dm
    )
    logger.info("Preparing data...")
    dm.prepare_data()
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        **cfg.finetune.training
    )
    logging.info("Starting training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    turngpt_finetune()