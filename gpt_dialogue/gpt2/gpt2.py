# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-11 15:55:27
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-18 08:40:19

import sys
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
from gpt_dialogue.gpt2.dm import GPT2FinetuneDM

from typing import Dict

import logging

logger = logging.getLogger(__name__)


class GPT2(LanguageModel):
    """
    LanguageModel subclass that acts as a standardized wrapper on top of GPT2.
    Underlying model details: https://huggingface.co/gpt2
    """

    _SUPPORTED_MODELS = ("gpt2", "gpt2-large", "distillgpt2")

    def __init__(self):
        self.model = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        return self._tokenizer

    def load(
        self,
        model_checkpoint: str = "gpt2",
        tokenizer_checkpoint: str = "gpt2",
        tokenizer_pad_token: str = None,
        tokenizer_eos_token: str = None,
        tokenizer_additional_special_tokens: List[str] = None,
        tokenizer_kwargs: Dict = {},
        model_kwargs: Dict = {},
    ):
        """
        Load the language model with the specified configs.

        Parameters
        ----------
        model_checkpoint : str, optional
            Can be gpt2, gpt2-large, distillgpt2, or a path to a pretrained
            model, by default "gpt2"
        tokenizer_checkpoint : str, optional
            Tokenizer ckpt path or name, by default "gpt2"
        tokenizer_pad_token : str, optional
            Pad token to be used. If unspecified, uses default settings,
            by default None
        tokenizer_eos_token : str, optional
            EOS token to be used. If unspecified, uses default, by default None
        tokenizer_additional_special_tokens : List[str], optional
            List of additional tokens in the dataset., by default None
        tokenizer_kwargs : Dict, optional
            Dictionary of args that can be used by huggingface tokenizers,
            by default {}
        model_kwargs : Dict, optional
            Dictionary of kwargs that can be used by huggingface models,
            by default {}
        """
        # Verify that the model is supported or that the path exists
        if (
            not os.path.exists(model_checkpoint)
            and not model_checkpoint in self._SUPPORTED_MODELS
        ):
            logger.error(
                "ERROR: Provide a valid path or selected pre-trained "
                f"models from {self._SUPPORTED_MODELS}"
            )
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint

        # Create the tokenizer args
        tokenizer_args = {
            "pretrained_model_name_or_path": tokenizer_checkpoint,
            "pad_token": tokenizer_pad_token,
            "eos_token": tokenizer_eos_token,
            "additional_special_tokens": tokenizer_additional_special_tokens,
        }
        tokenizer_args.update(tokenizer_kwargs)
        # Remove all the None args
        tokenizer_args = {k: v for k, v in tokenizer_args.items() if v != None}
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
        # Create the model args
        model_args = {
            "pretrained_model_name_or_path": model_checkpoint,
        }
        if tokenizer_pad_token != None:
            model_args["pad_token_id"] = self._tokenizer.pad_token_id
        if tokenizer_eos_token != None:
            model_args["eos_token_id"] = self._tokenizer.eos_token_id
        model_args.update(model_kwargs)
        # Remove all the None args
        model_args = {k: v for k, v in model_args.items() if v != None}
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(**model_args)
        self.model.resize_token_embeddings(len(self._tokenizer))

        # If there is only pretrained_model_name_or_path arg.
        if (
            len(model_args) == 1
            and len(tokenizer_args) == 1
            and model_args["pretrained_model_name_or_path"]
            in self._SUPPORTED_MODELS
            and tokenizer_args["pretrained_model_name_or_path"]
            in self._SUPPORTED_MODELS
        ):
            logger.warning(
                f"WARNING: Loading pretrained model {model_checkpoint} "
                f"and tokenizer {tokenizer_checkpoint} with "
                f"default arguments.\nThis is equivalent to using the specified "
                f"model and tokenizer directly from the transformers library."
            )

    def finetune(
        self,
        save_dir: str,
        train_csv_path: str,
        val_csv_path: str,
        data_block_size: int = 128,
        utterance_key: str = "Utterance",
        num_train_epochs: int = 30,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        warmup_steps: int = 300,
        report_to="wandb",
    ):
        """
        Fintune the model on the provided dataset.

        Parameters
        ----------
        save_dir : str
            Output directory where the finetuned models are saved.
        train_csv_path : str
            Path to a single csv that contains the training data.
        val_csv_path : str
        Path to a single csv that contains the validation data.
        data_block_size : int, optional
            Size by which the data is chunked in the finetuned data module,
            by default 128
        utterance_key : str, optional
            Column in the csv file that contains the unique
            utterance ids, by default "Utterance"
        num_train_epochs : int, optional
            Maximum number of training epochs, by default 30
        per_device_train_batch_size : int, optional
            Training batch size per device, by default 8
        per_device_eval_batch_size : int, optional
            Validation batch size per device, by default 8
        warmup_steps : int, optional
            Number of warmup steps for the optimizer, by default 300
        report_to : str, optional
            List of integrations to report the results and logs to, by default "wandb"
        """
        # Load the data modules / apply the data transforms
        dm = GPT2FinetuneDM(
            tokenizer=self._tokenizer,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            utterance_key=utterance_key,
            data_block_size=data_block_size,
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
            logging_steps=1,
            logging_dir=os.path.join(save_dir, "trainer_logs"),
            report_to=report_to,
        )

        # Create the trainer and start training
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
        )
        torch.cuda.empty_cache()
        gc.collect()
        trainer.train()
        trainer.save_model()

    def __call__(self, *args, **kwargs):
        """
        Apply the __call__ method for the underlying gpt2 model head,
        which in this case is GPT2LMHeadModel
        Link: https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/gpt2#transformers.GPT2Model
        """
        return self.model(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Base model: {self.model}\n" f"Base tokenizer: {self._tokenizer}"
        )

    def to(self, device: torch.device):
        """
        Set the model to the provided device.

        Parameters
        ----------
        device : torch.device
            Can be one of torch.device('cpu') or torch.device('gpu')
        """
        self.model.to(device)

    def eval(self):
        """
        Set the model to eval mode, which is required when using the model
        to make inferences
        """
        self.model.eval()

    def encode(self, text: Union[str, List], *args, **kwargs) -> List:
        """
        Encodes the given text for inference with this model using the underlying
        tokenizer.

        Parameters
        ----------
        text : Union[str, List]
            String of lists of strings to be encoded

        Returns
        -------
        List
           List containing the tokenized data.

        Raises
        ------
        NotImplementedError
            Raised if an input type is passed that is not supported.
        """
        if isinstance(text, list):
            text = " ".join(text)
            return self.tokenizer(text, *args, **kwargs)
        elif isinstance(text, str):
            return self.tokenizer(text, *args, **kwargs)
        else:
            raise NotImplementedError()

    def decode(self, input_ids: List, *args, **kwargs) -> str:
        """
        Converts a given list of tokens to a string.

        Parameters
        ----------
        input_ids : List
            ids that were generated using the encode method

        Returns
        -------
        str
            string that is represented by the provided ids.
        """
        return self._tokenizer.decode(input_ids, *args, **kwargs)
