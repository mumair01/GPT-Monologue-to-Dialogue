# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-06 15:31:31
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-05 11:52:44


import sys
import os
import argparse
import random
from dataclasses import dataclass
from datetime import datetime
import yaml
from tqdm import tqdm
from typing import Union, List

from omegaconf import DictConfig, OmegaConf
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

import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -----------------------=--- GLOBALS --------------------------------------

TOKENIZER_PAD_TOKEN = "<PAD>"
TOKENIZER_EOS_TOKEN = "<|endoftext|>"

# --  Set environment global vars.

CUDA_ENV = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda') if CUDA_ENV else torch.device('cpu')


# NOTE: Assuming that the dataset is in the correct format.
def load_inference_dataset(csv_path, start_conv_no, end_conv_no):
    df = pd.read_csv(csv_path,index_col=0)
    conversation_dfs = [df.loc[df['convID'] == i] for i in range(
        np.max(df['convID'].unique()) + 1)]
    if end_conv_no > len(conversation_dfs) or end_conv_no == -1:
        end_conv_no = len(conversation_dfs)
    assert len(conversation_dfs) >= end_conv_no
    assert start_conv_no < end_conv_no
    conversation_dfs = conversation_dfs[start_conv_no:end_conv_no]
    return conversation_dfs

# ------------------------ INFERENCE HELPERS  -----------------------------


def get_last_word_prob(model, tokenizer, text):
    sentence_so_far = text
    context = ' '.join(text.split()[:-1])
    # Encode
    context_encoding = tokenizer.encode(
        context, return_tensors="pt")
    whole_text_encoding = tokenizer.encode(
        sentence_so_far, return_tensors="pt")
    cw_encoding = whole_text_encoding[:, context_encoding.shape[1]:]
    # move to the appropriate device before inference
    whole_text_encoding = whole_text_encoding.to(TORCH_DEVICE)
    output = model(whole_text_encoding)
    # Obtain the logits for the last hidden state and the logits
    # that provide values for the tokens in the critical word.
    # i.e., if cw token starts at position i in the sentence, then the logits
    # are from i-1 to len(tokens) - 1.
    cw_extracted_logits = output.logits[-1, context_encoding.shape[1]-1:-1, :]
    # Obtain the probabilities from the logits
    softmax = torch.nn.Softmax(dim=-1)
    cw_extracted_probs_from_logits = softmax(cw_extracted_logits)
    # NOTE: Converting to log scale and taking exponential sum of the log
    # probabilities at the end will ensure that there is not floating point
    # overflow issue for very small probability values.
    cw_extracted_log_probs_from_logits = torch.log(
        cw_extracted_probs_from_logits)
    # Extract the probabilities of the specific tokens
    cw_tokens_probs = []
    for cw_subtoken, probs in zip(cw_encoding[0], cw_extracted_log_probs_from_logits):
        cw_tokens_probs.append(probs[cw_subtoken])
    return float(torch.exp(torch.sum(torch.Tensor(cw_tokens_probs))))

def get_final_n_word_probs(model, tokenizer, text,
        N, context_buffer_size):
    words = text.strip().split(' ')
    if N == -1:
        N = len(words)
    assert not (N > len(words) or N<= 0)
    words[:len(words) - N]
    sentence_so_far = " ".join(words[:len(words) - N])
    results = []
    for word in words[len(words) - N:]:
        sentence_so_far += " " + word.strip()
        # Reset the buffer if required
        num_words_so_far = len(sentence_so_far.split(' '))
        if num_words_so_far > context_buffer_size:
            sentence_so_far = " ".join(
                sentence_so_far.split(' ')[num_words_so_far - context_buffer_size - 1:])
        last_word_prob = get_last_word_prob(
            model, tokenizer, sentence_so_far)
        context = " ".join(sentence_so_far.split(' ')[:-1])
        results.append((context,word,last_word_prob))
    return np.asarray(results)


def generate_conditional_probs(model, tokenizer, conversation_df,
        N, context_buffer_size, conv_no):
    results_list = []
    text = ""
    pbar = tqdm(desc="Processing conversation {}".format(conv_no),
                    total=len(conversation_df))
    for turn_no, turn in enumerate(conversation_df.itertuples()):
        text += " " +  turn.Utterance.strip()
        text = text.strip()
        turn_length = len(turn.Utterance.split(' '))
        n_probs = turn_length if N == -1 or N > turn_length else N
        results = get_final_n_word_probs(
            model, tokenizer,text,n_probs, context_buffer_size)
        for result_no, result in enumerate(results):
            word_no = turn_length - n_probs + result_no
            context, word, prob = result
            results_list.append((
                turn.convName,turn.convID, turn_no, word_no, context,word,prob))
        pbar.update()
    return results_list


# ------------------------ INFERENCE MAIN  --------------------------------

def surprisal_inference(
        model_checkpoint : str,
        tokenizer_checkpoint : str,
        tokenizer_additional_special_tokens : List[str],
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
        pad_token=TOKENIZER_PAD_TOKEN,
        eos_token=TOKENIZER_EOS_TOKEN,
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
        pd.DataFrame(results,columns=df_columns).to_csv(
            os.path.join(save_dir,
                "{}_conditional_probs.csv".format(results[0][0])))
        data.extend(results)
    # Save the results as a dataframe
    results_df = pd.DataFrame(data, columns=df_columns)
    logger.info("Saving results")
    results_df.to_csv(
        os.path.join(save_dir,"conditional_probs_combined.csv"))
    logger.info("Complete!")



@hydra.main(version_base=None, config_path="../../conf", config_name="inference_transformers")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
     # -- Create loggers for this script
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(cfg.paths.save_dir,"inference.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Parse the dataset
    surprisal_inference(
        model_checkpoint=cfg.model.model_checkpoint,
        tokenizer_checkpoint=cfg.model.tokenizer_checkpoint,
        tokenizer_additional_special_tokens=cfg.model.tokenizer_additional_special_tokens,
        dataset_path=cfg.dataset_path,
        save_dir = cfg.paths.save_dir,
        start_conversation_no=cfg.inference.start_conversation_no,
        end_conversation_no=cfg.inference.end_conversation_no,
        n_probs=cfg.inference.n_probs,
        context_buffer_size=cfg.inference.context_buffer_size
    )


if __name__ == "__main__":
    main()




