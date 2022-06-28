# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-06-20 09:02:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-06-27 15:56:25

from cgitb import reset
import sys
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from dataclasses import dataclass

import random
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


# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# -------------------- ENV. VARS. ------------------------

# --  Set environment global vars.

# Shared env. vars.
GLOBAL_SEED = 42
IS_CUDA_ENV = torch.cuda.is_available()
GLOBAL_DEVICE = torch.device('cuda') if IS_CUDA_ENV else torch.device('cpu')
SET_SEED = True # If true, sets the global seed.
# If true, assumes that there are limited resources available and uses limited data / models.
LIMITED_RESOURCES = not IS_CUDA_ENV

# Configuring env.
if SET_SEED:
    # to make this notebook's output stable across runs
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

# --- Other vars.

# NOTE: The below should be the same in the dataset - assuming there are 2 speakers!
SPEAKER_1_TOKEN = "<SP1>"
SPEAKER_2_TOKEN = "<SP2>"
CONV_START_TOKEN = "<START>"
CONV_END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<|endoftext|>"

# Tokenizer vars.
TOKENIZER_CHECKPOINT = "gpt2"
TOKENIZER_BATCH_SIZE = 128

# Custom dataset vars.
CUSTOM_DATASET_CHUNK_SIZE = 32

# Model vars.
MODEL_CHECKPOINT = "distilgpt2" if LIMITED_RESOURCES else "gpt2-large"


# -------------------- HELPER METHODS --------------------

@dataclass
class Configs:
    root : str
    # Data paths
    dataset_name : str
    dataset_type : str
    train_path : str
    val_path : str
    # Save paths
    save_model_dir : str
    reports_dir : str
    # Dataset args
    custom_dataset : bool
    # Training args
    num_train_epochs : int
    per_device_train_batch_size : int
    per_device_eval_batch_size : int
    eval_steps : int
    # save_steps : int
    warmup_steps : int


def reset_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def config_env(config_path):
    print("Loading configurations from path: {}".format(config_path))
    with open(config_path,"r") as f:
        configs = yaml.safe_load(f)
        pprint.pprint(configs)
        configs = Configs(
            configs['root'],
            configs["dataset"]["name"],
            configs["dataset"]["type"],
            os.path.join(configs['root'],configs["dataset"]["train_path"]),
            os.path.join(configs["root"],configs["dataset"]["train_path"]),
            os.path.join(configs["root"],configs["results"]["save_model_dir"]),
            os.path.join(configs["root"],configs["results"]["reports_dir"]),
            # Dataset args
            configs["dataset"]["custom"],
            # Training args
            configs["training"]["num_train_epochs"],
            configs["training"]["per_device_train_batch_size"],
            configs["training"]["per_device_eval_batch_size"],
            configs["training"]["eval_steps"],
            # configs["training"]["save_steps"],
            configs["training"]["warmup_steps"])
        # Check dirs.
        reset_dir(configs.save_model_dir)
        reset_dir(configs.reports_dir)
        assert os.path.isfile(configs.train_path)
        assert os.path.isfile(configs.val_path)
        return configs

# --- Custom Dataset methods

def tokenize_fn(tokenizer):
    return lambda data: tokenizer(data["Utterance"], truncation=True)

def chunk_tokenized_samples(tokenized_samples,chunk_size=CUSTOM_DATASET_CHUNK_SIZE):
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
    print("Starting finetuning using device {}...".format(GLOBAL_DEVICE))
    # Load the tokenizer with special tokens defined.
    print("Loading tokenizer: {}".format(TOKENIZER_CHECKPOINT))
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_CHECKPOINT,
        pad_token = PAD_TOKEN,
        eos_token = EOS_TOKEN,
        additional_special_tokens=(
            SPEAKER_1_TOKEN, SPEAKER_2_TOKEN, CONV_START_TOKEN,
            CONV_END_TOKEN))
    # Save the tokenizer after adding new tokens
    tokenizer.save_pretrained(configs.save_model_dir)
    # Create the data collator, which is responsible for creating batches from the
    # datasets during training.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt")
    # Loading the dataset based on the configuration.
    if configs.custom_dataset:
        print("Loading custom dataset...")
        dataset = load_dataset("csv", data_files={
            'train' : configs.train_path,
            'validation' : configs.val_path})
        # Once loaded, the dataset needs to be processed
        tokenized_datasets = dataset.map(
            tokenize_fn(tokenizer), batched=True,
            batch_size=TOKENIZER_BATCH_SIZE,
            remove_columns=["Unnamed: 0","convID","Utterance"])
        lm_datasets = tokenized_datasets.map(
            chunk_tokenized_samples,batched=True)
    else:
        print("Loading TextDataset...")
        lm_datasets = {}
        for name, path in zip(('train','validation'),(configs.train_path, configs.val_path)):
            lm_datasets[name] = TextDataset(
                tokenizer=tokenizer,
                file_path=path,
                block_size= 128 # TODO: Make this configurable variable later.
            )
    # Load the model
    print("Loading model: {}".format(MODEL_CHECKPOINT))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
    )
    model.resize_token_embeddings(len(tokenizer))
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()
    # Create training args and train
    # Defining training arguments
    print("Preparing training arguments...")
    training_args = TrainingArguments(
            output_dir=configs.save_model_dir,
            overwrite_output_dir=True,
            num_train_epochs=configs.num_train_epochs,
            per_device_train_batch_size=configs.per_device_train_batch_size,
            per_device_eval_batch_size=configs.per_device_eval_batch_size,
            # eval_steps=configs.eval_steps,
            save_strategy="epoch",
            # save_steps=configs.save_steps,
            warmup_steps=configs.warmup_steps,
            prediction_loss_only=True,
            evaluation_strategy='epoch',
            gradient_accumulation_steps=2,
            eval_accumulation_steps=1
            # logging_dir=configs.reports_dir
        )
    # Create the trainer
    # NOTE: Trainer should automatically put the model and dataset to GPU
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_datasets['train'],
        eval_dataset=lm_datasets['validation'])
    print("Starting training...")
    trainer.train()
    print("Saving trained model...")
    trainer.save_model()
    print("Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",type=str,required=True, help="Configuration file path")
    args = parser.parse_args()
    # Load the configuration file and parse it
    configs = config_env(args.config)
    finetune(configs)







