# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 15:21:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-24 12:22:22

##############################
# This script contains the loader, trainer, and predictor for TurnGPT.
##############################

import os
import sys
import torch
from typing import Dict, Optional, List, Any


import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from .model import TurnGPTDoubleHeadsModel, TurnGPTLMHeadModel
from LLMDialogue.models import LanguageModel
from LLMDialogue.dm.spokenLanguageDM import SpokenLanguageModelingDM
from LLMDialogue.utils.general import remove_keys_from_dict


class TurnGPT(LanguageModel):
    _MODEL_MAP = {
        "LMHead": TurnGPTLMHeadModel,
        "DoubleHeads": TurnGPTDoubleHeadsModel,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        # Attributes.
        self._model = None
        self._tokenizer = None
        self.pretrained_model_name_or_path = None
        self.model_head = None
        self.device = "auto"

    def load(
        self,
        pretrained_model_name_or_path: str = "gpt2",
        model_head: str = "LMHead",
        **kwargs,
    ):
        keys = ("pretrained_model_name_or_path", "model_head")
        assert all([k not in kwargs for k in keys])

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_head = model_head

        if model_head not in self._MODEL_MAP.keys():
            raise Exception(
                f"TurnGPT with head {model_head} not supported - use one of {list(self._MODEL_MAP.keys())}"
            )

        if os.path.isfile(pretrained_model_name_or_path):
            self._model = self._MODEL_MAP[model_head].load_from_checkpoint(
                pretrained_model_name_or_path
            )
        else:
            self._model = self._MODEL_MAP[model_head](
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                **kwargs,
            )
        # Set the tokenizer
        self._tokenizer = self._model.tokenizer

    def finetune(
        self,
        dm: SpokenLanguageModelingDM,
        save_dir: str,
        max_epochs: int = 30,
        log_every_n_steps: int = 1,
        report_to: str = None,
        **kwargs,
    ):
        keys = ("dm", "save_dir", "max_epochs", "log_every_n_steps", "report_to")
        assert all([k not in kwargs for k in keys])

        assert isinstance(dm, SpokenLanguageModelingDM)
        # Prepare the output directory.
        os.makedirs(save_dir, exist_ok=True)

        # Disable tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Load the data module
        dm.prepare_data()
        dm.setup(stage="fit")

        # Create the logger
        training_logger = self._create_training_logger(
            report_to=report_to, save_dir=save_dir
        )

        # Create the callbacks
        callbacks = self._create_callbacks()

        # Define the trainer

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=self.device.type if self.device is not None else "auto",
            logger=training_logger,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks,
            **kwargs,
        )
        trainer.fit(self.model, datamodule=dm)

    def encode(self, text, *args, **kwargs):
        """
        Encode text as required for inference by this model.
        NOTE: Ensure that the args passed to the tokenizer are correct.
        """
        return self._tokenizer(text, add_eos_token=False * args, **kwargs)

    def decode(self, input_ids, *args, **kwargs):
        return self._tokenizer.decode(input_ids, *args, **kwargs)

    def _create_training_logger(self, report_to: str, save_dir: str):
        # Create CSV logger
        model_name = f"TurnGPT_{self.model_head}"
        if report_to == "none" or report_to == "csv":
            training_logger = CSVLogger(
                save_dir=save_dir,
                name=model_name,
            )
        elif report_to == "wandb":
            training_logger = WandbLogger(
                name=model_name, save_dir=save_dir, log_model=True
            )
        return training_logger

    def _create_callbacks(self) -> List[Callback]:
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            # NOTE: We want to save the models at every epoch.
            save_top_k=-1,
            save_last=True,
        )

        return [checkpoint_callback]
