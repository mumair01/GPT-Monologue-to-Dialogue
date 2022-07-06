# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-06-20 09:02:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-06 10:45:20

import sys
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from copy import deepcopy
from typing import Union, List
from datetime import datetime
from functools import partial


import pprint
import random
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


# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import logging
# Enable all level loggers for this script.
logging.getLogger(__name__).setLevel(logging.NOTSET)


# -------------------- ENV. VARS. ------------------------

TOKENIZER_PAD_TOKEN = "<PAD>"
TOKENIZER_EOS_TOKEN = "<|endoftext|>"

# --  Set environment global vars.

CUDA_ENV = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda') if CUDA_ENV else torch.device('cpu')
# -------------------- HELPER METHODS --------------------

@dataclass
class Configs:

    @dataclass
    class Env:
        seed : Union[int, None]
    @dataclass
    class Dataset:
        dataset_name : str
        train_path : str
        val_path : str
        custom_dataset : bool
    @dataclass
    class Results:
        # Save paths
        save_dir : str
        reports_dir : str
    @dataclass
    class Training:
        # Training args
        model_checkpoint : str
        tokenizer_checkpoint : str
        tokenizer_additional_special_tokens : List[str]
        data_block_size : int
        num_train_epochs : int
        per_device_train_batch_size : int
        per_device_eval_batch_size : int
        warmup_steps : int

    root : str
    name : str
    env : Env
    dataset : Dataset
    results : Results
    training : Training


def parse_configs(config_path):
    with open(config_path,"r") as f:
        configs_data = yaml.safe_load(f)
        pprint.pprint(configs_data)
        # Obtaining timestamp for output.
        now = datetime.now()
        ts = now.strftime('%m-%d-%Y_%H-%M-%S')
        # Creating configs
        configs = Configs(
            root=configs_data['root'],
            name=configs_data["dataset"]["name"],
            env=Configs.Env(
                seed=configs_data['env']['seed']
            ),
            dataset=Configs.Dataset(
                dataset_name=configs_data['dataset']['name'],
                train_path=os.path.join(configs_data['root'],configs_data["dataset"]["train_path"]),
                val_path=os.path.join(configs_data["root"],configs_data["dataset"]["val_path"]),
                custom_dataset=configs_data['dataset']['custom']
            ),
            results=Configs.Results(
                save_dir=os.path.join(configs_data["root"],configs_data["results"]["save_dir"],ts),
                reports_dir=os.path.join(configs_data["root"],configs_data["results"]["reports_dir"],ts),
            ),
            training=Configs.Training(
                model_checkpoint=configs_data["training"]["model_checkpoint"],
                tokenizer_checkpoint=configs_data["training"]["tokenizer_checkpoint"],
                tokenizer_additional_special_tokens=configs_data['training']['tokenizer_additional_special_tokens'],
                data_block_size=configs_data['training']['data_block_size'],
                num_train_epochs=configs_data["training"]["num_train_epochs"],
                per_device_train_batch_size=configs_data["training"]["per_device_train_batch_size"],
                per_device_eval_batch_size=configs_data["training"]["per_device_eval_batch_size"],
                warmup_steps=configs_data["training"]["warmup_steps"])
            )
        return configs

def config_env(config_path):
    # Parse configs
    logging.info("Loading configurations from path: {}".format(config_path))
    configs = parse_configs(config_path)
    # Set seed if required
    if configs.env.seed != None:
        np.random.seed(configs.env.seed)
        torch.manual_seed(configs.env.seed)
    # Check input files exist
    assert os.path.isfile(configs.dataset.train_path)
    assert os.path.isfile(configs.dataset.val_path)
    # Create output directories
    os.makedirs(configs.results.save_dir)
    os.makedirs(configs.results.reports_dir,exist_ok=True)
    return configs

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


# -------------------- MAIN METHODS ----------------------


def finetune(configs : Configs):
    logging.info("Starting finetuning using device {}...".format(TORCH_DEVICE))
    # Load the tokenizer with special tokens defined.
    logging.info("Loading tokenizer: {}".format(configs.training.tokenizer_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(
        configs.training.tokenizer_checkpoint,
        pad_token=TOKENIZER_PAD_TOKEN,
        eos_token=TOKENIZER_EOS_TOKEN,
        additional_special_tokens=
            configs.training.tokenizer_additional_special_tokens)
    # Save the tokenizer after adding new tokens in a separate dir.
    tokenizer_save_dir = os.path.join(configs.results.save_dir,"tokenizer")
    tokenizer.save_pretrained(tokenizer_save_dir)
    logging.info("Using tokenizer:\n{}".format(tokenizer))
    # Loading the dataset based on the configuration.
    if configs.dataset.custom_dataset:
        logging.info("Loading data as custom dataset...")
        logging.warn("Custom dataset expects .csv files.")
        dataset = load_dataset("csv", data_files={
            'train' : configs.dataset.train_path,
            'validation' : configs.dataset.val_path})
        # Once loaded, the dataset needs to be processed
        tokenized_datasets = dataset.map(
            tokenize_fn(tokenizer), batched=True,
            remove_columns=["Unnamed: 0","convID","Utterance"])
        lm_datasets = tokenized_datasets.map(
            partial(chunk_tokenized_samples,
                chunk_size=configs.training.data_block_size),batched=True)
    else:
        logging.info("Loading data as TextDataset...")
        logging.warn("TextDataset expects .txt files.")
        lm_datasets = {}
        for name, path in zip(('train','validation'),
                (configs.dataset.train_path, configs.dataset.val_path)):
            # TODO: Determine if I need this deepcopy.
            lm_datasets[name] = deepcopy(TextDataset(
                tokenizer=tokenizer,
                file_path=path,
                block_size= configs.training.data_block_size
            ))
    logging.info("Tokenizer length after loading datasets: {}".format(len(tokenizer)))
    # Create the data collator, which is responsible for creating batches from the
    # datasets during training.
    logging.info("Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt")
    # Load the model
    logging.info("Loading model: {}".format(configs.training.model_checkpoint))
    model = AutoModelForCausalLM.from_pretrained(
        configs.training.model_checkpoint,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
    )
    logging.info("Resizing model embeddings to {}".format(len(tokenizer)))
    model.resize_token_embeddings(len(tokenizer))
    # Create training args and train
    # Defining training arguments
    logging.info("Preparing training arguments...")
    training_args = TrainingArguments(
            output_dir = os.path.join(configs.results.save_dir,"trainer"),
            overwrite_output_dir=True,
            num_train_epochs=configs.training.num_train_epochs,
            per_device_train_batch_size=configs.training.per_device_train_batch_size,
            per_device_eval_batch_size=configs.training.per_device_eval_batch_size,
            warmup_steps=configs.training.warmup_steps,
            prediction_loss_only=True,
            save_strategy="epoch",
            evaluation_strategy='epoch',
            logging_strategy="epoch",
            logging_dir=os.path.join(configs.results.reports_dir,"trainer_logs")
        )
    # Create the trainer
    # NOTE: Trainer should automatically put the model and dataset to GPU
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_datasets['train'],
        eval_dataset=lm_datasets['validation'])
    logging.info("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Starting training...")
    trainer.train()
    logging.info("Saving trained model...")
    trainer.save_model()
    logging.info("Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",type=str,required=True, help="Configuration file path")
    args = parser.parse_args()
    # Load the configuration file and parse it
    configs = config_env(args.config)
    finetune(configs)







