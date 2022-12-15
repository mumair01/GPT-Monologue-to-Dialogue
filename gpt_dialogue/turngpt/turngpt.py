# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-11 15:54:22
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-15 15:57:41

##############################
# This script contains the loader, trainer, and predictor for TurnGPT.
##############################

import os
import sys
from typing import List

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from gpt_dialogue.turngpt.model import (
    TurnGPTDoubleHeadsModel,
    TurnGPTLMHeadModel
)

from gpt_dialogue.model import LanguageModel
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM

class TurnGPT(LanguageModel):

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
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
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
        # Disable tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        # training_logger = CSVLogger(
        #     save_dir=os.getcwd(),
        #     name=model_name)
        training_logger = WandbLogger(
            name=model_name,
            save_dir=os.getcwd(),
            log_model=True
        )

        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            # NOTE: We want to save the models at every epoch.
            save_top_k=-1
        )

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=training_logger,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            callbacks=[
                checkpoint_callback
            ],
            **kwargs
        )
        trainer.fit(self.model,datamodule=dm)

    def __repr__(self):
        return (
            f"Base model: {self.model}\n"
            f"Base tokenizer: {self.tokenizer}"
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def encode(self, text, *args, **kwargs):
        """
        Encode text as required for inference by this model.
        NOTE: Ensure that the args passed to the tokenizer are correct.
        """
        return self.tokenizer(
            text,
            add_eos_token=False
            *args, **kwargs
        )

    def decode(self, input_ids, *args, **kwargs):
        return self.tokenizer.decode(
            input_ids, *args, **kwargs
        )


