# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-06 15:31:31
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-11 11:35:30


import sys
import os
import argparse
import random
from dataclasses import dataclass
from datetime import datetime
import yaml
from tqdm import tqdm
from typing import Union, List

from omegaconf import DictConfig
import hydra

import pandas as pd
import numpy as np
import torch
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


import transformers
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelForCausalLM
from transformers import AutoTokenizer

from gpt_dialogue.scripts.inference.utils import (
    load_inference_dataset,
    generate_conditional_probs
)

import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -----------------------=--- GLOBALS --------------------------------------

# --  Set environment global vars.

HYDRA_CONFIG_RELATIVE_PATH = "../../../conf"
HYDRA_CONFIG_NAME = "inference"
CUDA_ENV = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda') if CUDA_ENV else torch.device('cpu')

# ------------------------ INFERENCE MAIN  --------------------------------

def surprisal_inference(
        model_checkpoint : str,
        tokenizer_checkpoint : str,
        tokenizer_additional_special_tokens : List[str],
        tokenizer_pad_token : str,
        tokenizer_eos_token : str,
        dataset_path : str,
        save_dir : str,
        start_conversation_no : int,
        end_conversation_no : int,
        n_probs : int,
        context_buffer_size : int,
    ):
    # Load the inference dataset
    logger.info("Loading inference dataset: {}".format(dataset_path))
    conversation_dfs = load_inference_dataset(
        csv_path=dataset_path,
        start_conv_no=start_conversation_no,
        end_conv_no=end_conversation_no)
    # Load the tokenizer
    logger.info("Loading tokenizer checkpoint: {}".format(tokenizer_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_checkpoint,
        pad_token=tokenizer_pad_token,
        eos_token=tokenizer_eos_token,
        additional_special_tokens=tokenizer_additional_special_tokens)
    # Save the tokenizer after adding new tokens in a separate dir.
    tokenizer_save_dir = os.path.join(save_dir,"tokenizer")
    logger.info("Saving tokenizer: {}".format(tokenizer_save_dir))
    tokenizer.save_pretrained(tokenizer_save_dir)
    # Load the model
    logger.info("Loading model checkpoint: {}".format(model_checkpoint))
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(TORCH_DEVICE) # NOTE: Moving to GPU works only for single-GPU inference.
    # Generate conditional probabilities for each conversation_df
    logger.info("Generating conditional probabilities")
    data = []
    df_columns = [
        'conversationName','conversationNumber', 'turnNumber','wordNumber','context',
        'word', 'probability']
    for i, conversation_df in enumerate(conversation_dfs):
        results = generate_conditional_probs(
            model=model,
            tokenizer=tokenizer,
            conversation_df=conversation_df,
            N=n_probs,
            context_buffer_size=context_buffer_size,
            conv_no=i)
        # Load the data as a single dataframe and save (important if the
        # program crashes).
        save_path = os.path.join(save_dir,
                "{}_conditional_probs.csv".format(results[0][0]))
        pd.DataFrame(results,columns=df_columns).to_csv(save_path)
        logger.info(f"Saving results: {save_path}")
        data.extend(results)
    # Save the results as a dataframe
    results_df = pd.DataFrame(data, columns=df_columns)
    logger.info("Saving results")
    results_df.to_csv(
        os.path.join(save_dir,"conditional_probs_combined.csv"))
    logger.info("Complete!")



@hydra.main(version_base=None, config_path=HYDRA_CONFIG_RELATIVE_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg : DictConfig):
    """
    Runs script as a hydra app.
    NOTE: This requires a +env and +dataset argument with the run command.
    Ex: python inference_transformers.py +env=local +dataset=inference/icc
    """
    # Parse the configs and run.
    surprisal_inference(
        model_checkpoint=cfg.inference.model.model_checkpoint,
        tokenizer_checkpoint=cfg.inference.model.tokenizer_checkpoint,
        tokenizer_additional_special_tokens=list(cfg.inference.model.tokenizer_additional_special_tokens),
        tokenizer_pad_token=cfg.inference.model.tokenizer_pad_token,
        tokenizer_eos_token=cfg.inference.model.tokenizer_eos_token,
        dataset_path=os.path.join(cfg.env.paths.root,cfg.dataset.test_path),
        save_dir=os.getcwd(),
        start_conversation_no=cfg.inference.start_conversation_no,
        end_conversation_no=cfg.inference.end_conversation_no,
        n_probs=cfg.inference.n_probs,
        context_buffer_size=cfg.inference.context_buffer_size
    )

if __name__ == "__main__":
    main()




