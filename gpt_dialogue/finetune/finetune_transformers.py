# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-06-20 09:02:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-04 14:35:10

import os
from typing import Union, List
from datetime import datetime
from functools import partial


from omegaconf import DictConfig, OmegaConf
import hydra
import pprint
import gc
import shutil
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Others
import glob

# Transformers
import transformers
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset



import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# -------------------- ENV. VARS. ------------------------

TOKENIZER_PAD_TOKEN = "<PAD>"
TOKENIZER_EOS_TOKEN = "<|endoftext|>"

# --  Set environment global vars.

CUDA_ENV = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda') if CUDA_ENV else torch.device('cpu')


# --- Custom Dataset methods

def tokenize_fn(tokenizer):
    return lambda data: tokenizer(data["Utterance"], truncation=True)

def chunk_tokenized_samples(tokenized_samples,chunk_size):
    # Concatenate all the utterances
    keys =  ('input_ids','attention_mask')
    concatenated_examples = {k : sum(tokenized_samples[k],[]) for k in keys}
    total_length = len(concatenated_examples[keys[0]])
    total_length = (total_length // chunk_size) * chunk_size
    chunks = {
        k : [concatenated_examples[k][i:i+ chunk_size] \
            for i in range(0, total_length,chunk_size)]  for k in keys
    }
    return chunks


def transformers_finetune(
        model_checkpoint : str,
        tokenizer_checkpoint : str,
        tokenizer_additional_special_tokens : List[str],
        save_dir : str,
        train_path : str,
        val_path : str,
        data_block_size,
        num_train_epochs,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        warmup_steps):
    logger.info("Using device {}".format(TORCH_DEVICE))
    # Load the tokenizer with special tokens defined.
    logger.info("Loading tokenizer: {}".format(tokenizer_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_checkpoint,
        pad_token=TOKENIZER_PAD_TOKEN,
        eos_token=TOKENIZER_EOS_TOKEN,
        additional_special_tokens=tokenizer_additional_special_tokens)
     # Save the tokenizer after adding new tokens in a separate dir.
    tokenizer_save_dir = os.path.join(save_dir,"tokenizer")
    tokenizer.save_pretrained(tokenizer_save_dir)
    logger.info("Loading data as custom dataset...")
    logger.warning("Custom dataset expects .csv files.")
    dataset = load_dataset("csv", data_files={
        'train' : train_path,
        'validation' : val_path})
    # Once loaded, the dataset needs to be processed
    tokenized_datasets = dataset.map(
        tokenize_fn(tokenizer), batched=True,
        remove_columns=["Unnamed: 0","convName","convID","Utterance"])
    lm_datasets = tokenized_datasets.map(
        partial(chunk_tokenized_samples,
            chunk_size=data_block_size),batched=True)
    logger.info("Tokenizer length after loading datasets: {}".format(len(tokenizer)))
    # Create the data collator, which is responsible for creating batches from the
    # datasets during training.
    logger.info("Creating data collator.")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt")
    # Load the model
    logger.info("Loading model: {}".format(model_checkpoint))
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
    )
    logger.info("Resizing model embeddings to {}".format(len(tokenizer)))
    model.resize_token_embeddings(len(tokenizer))
    # Create training args and train
    # Defining training arguments
    logger.info("Preparing training arguments.")
    training_args = TrainingArguments(
            output_dir = os.path.join(save_dir,"trainer"),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            prediction_loss_only=True,
            save_strategy="epoch",
            evaluation_strategy='epoch',
            logging_strategy="epoch",
            logging_dir=os.path.join(save_dir,"trainer_logs"),
            report_to="none"
        )
    # Create the trainer
    # NOTE: Trainer should automatically put the model and dataset to GPU
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_datasets['train'],
        eval_dataset=lm_datasets['validation'])
    logger.info("Clearing CUDA cache")
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving trained model...")
    trainer.save_model()
    logger.info("Completed!")


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_transformers")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Parse the dataset
    transformers_finetune(
        model_checkpoint=cfg.model.model_checkpoint,
        tokenizer_checkpoint=cfg.model.tokenizer_checkpoint,
        tokenizer_additional_special_tokens=list(cfg.model.tokenizer_additional_special_tokens),
        save_dir=cfg.paths.save_dir,
        train_path=cfg.dataset.train_path,
        val_path=cfg.dataset.validation_path,
        data_block_size=cfg.training.data_block_size,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        warmup_steps=cfg.training.warmup_steps
    )

if __name__ == "__main__":
    main()

