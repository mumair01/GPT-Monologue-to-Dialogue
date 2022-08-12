# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-11 15:54:22
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-12 13:46:21

##############################
# This script contains the loader, trainer, and predictor for TurnGPT.
##############################

import os
import sys
from typing import List

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from gpt_dialogue.turngpt.model import (
    TurnGPTDoubleHeadsModel,
    TurnGPTLMHeadModel
)
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM

class TurnGPT:

    _MODEL_MAP = {
        "LMHead" : TurnGPTLMHeadModel,
        "DoubleHeads" : TurnGPTDoubleHeadsModel
    }

    def __init__(self):
        self.model = None

    @property
    def tokenizer(self):
        return self.model._tokenizer

    def load(
        self,
        pretrained_model_name_or_path : str = "gpt2",
        model_head : str = "LMHead",
        **kwargs
    ):

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_head = model_head

        if model_head not in self._MODEL_MAP.keys():
            raise Exception(
                f"TurnGPT with head {model_head} not supported - use one of {list(self._MODEL_MAP.keys())}"
            )

        if os.path.isfile(pretrained_model_name_or_path):
            self.model = self._MODEL_MAP[model_head].load_from_checkpoint(
                pretrained_model_name_or_path)
        else:
            self.model = self._MODEL_MAP[model_head](
                    pretrained_model_name_or_path=pretrained_model_name_or_path
                    **kwargs
                )

    def finetune(
        self,
        # Data Module Args
        train_csv_path : str,
        val_csv_path : str,
        save_dir : str ,
        conversation_id_key : str = "convID",
        utterance_key : str = "Utterance",
        batch_size : int = 16,
        max_length : int = 1024,
        chunk_size : int = 128,
        num_workers : int = 8,
        pin_memory : int = False,
        use_cache : bool = False,
        # Training Args
        max_epochs : int = 30,
        log_every_n_steps : int = 1,
        **kwargs
    ):
        # Load the data module
        dm = TurnGPTFinetuneDM(
            tokenizer=self.model._tokenizer,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            conversation_id_key=conversation_id_key,
            utterance_key=utterance_key,
            save_dir=save_dir,
            batch_size=batch_size,
            max_length=max_length,
            chunk_size=chunk_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_cache=use_cache
        )
        # Create CSV logger
        model_name = f"TurnGPT_{self.model_head}"
        training_logger = CSVLogger(
            save_dir=os.getcwd(),
            name=model_name)
        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=training_logger,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            **kwargs
        )
        trainer.fit(self.model,datamodule=dm)

    def __call__(self, data):
        return self.model(data)
