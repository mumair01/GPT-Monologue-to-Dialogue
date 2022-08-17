# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-11 15:55:27
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 16:46:12

import os
from typing import Union, List

import gc
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Pytorch imports
import torch

# Transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

from gpt_dialogue.model import LanguageModel
from gpt_dialogue.monologue_gpt.dm import MonologueGPTFinetuneDM

class MonologueGPT(LanguageModel):

    _SUPPORTED_MODELS = (
        "gpt2",
        "gpt2-large"
    )

    _TOKENIZER_EOS_TOKEN = "<|endoftext|>"
    _TOKENIZER_PAD_TOKEN = "<PAD>"
    _TOKENIZER_ADDITIONAL_SPECIAL_TOKENS = [
        "<SP1>", # Speaker 1 token
        "<SP2>", # Speaker 2 token
        "<START>", # Conversation start token
        "<END>" # Conversation end token
    ]

    def __init__(self):
        self.model= None
        self._tokenizer = None

    @property
    def tokenizer(self):
        return self._tokenizer


    def load(
        self,
        model_checkpoint : str = "gpt2",
        tokenizer_checkpoint : str = "gpt2",
        tokenizer_pad_token :str = _TOKENIZER_PAD_TOKEN,
        tokenizer_eos_token : str = _TOKENIZER_EOS_TOKEN,
        tokenizer_additional_special_tokens : List[str] = _TOKENIZER_ADDITIONAL_SPECIAL_TOKENS,

    ):
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint,
            pad_token=tokenizer_pad_token,
            eos_token=tokenizer_eos_token,
            additional_special_tokens=tokenizer_additional_special_tokens)



        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            pad_token_id = self._tokenizer.pad_token_id,
            eos_token_id = self._tokenizer.eos_token_id
        )
        self.model.resize_token_embeddings(len(self._tokenizer))

    def finetune(
        self,
        save_dir : str,
        train_csv_path : str,
        val_csv_path : str ,
        data_block_size : int = 128,
        utterance_key : str="Utterance",
        num_train_epochs : int = 30,
        per_device_train_batch_size : int = 8,
        per_device_eval_batch_size : int = 8,
        warmup_steps : int = 300,
    ):

        print(save_dir, train_csv_path, val_csv_path)
        # Load the data modules / apply the data transforms
        dm = MonologueGPTFinetuneDM(
            tokenizer=self._tokenizer,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            utterance_key=utterance_key,
            data_block_size=data_block_size
        )
        data_collator, lm_datasets = dm()

        # Defining the Training Arguments
        training_args = TrainingArguments(
            output_dir=save_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            prediction_loss_only=True,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            logging_dir=os.path.join(save_dir,"trainer_logs"),
            report_to="none"
        )

        # Create the trainer and start training
        trainer  = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=lm_datasets['train'],
            eval_dataset=lm_datasets['validation']
        )
        torch.cuda.empty_cache()
        gc.collect()
        trainer.train()
        trainer.save_model()

    def __call__(self, data):
        return self.model(data)