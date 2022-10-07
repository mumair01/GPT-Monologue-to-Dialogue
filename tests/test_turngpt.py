# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-31 15:39:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-10-07 14:59:11

import pytest
import sys

from transformers import AutoTokenizer, GPT2LMHeadModel
import transformers
import pytorch_lightning as pl
import numpy as np

from gpt_dialogue.turngpt.tokenizer import SpokenNormalizer, SpokenDialogueTokenizer
from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM


def test_spoken_normalizer():
    normalizer = SpokenNormalizer()
    string = "Hello,how are you?"
    print(normalizer.normalize_str(string,add_whitespace_punc=True))
    print(normalizer.normalize_str(string,add_whitespace_punc=False))

def test_spoken_tokenizer():
    tokenizer = SpokenDialogueTokenizer("gpt2")
    # Case 1: Simple string
    print("Case 1.1")
    print(tokenizer.encode("Hello, how are you doing?"))
    print("Case 1.2")
    print(tokenizer("Hello, how are you doing?"))
    # Case 2: List of strings representing multiple turns in a single conversation.
    print("Case 2.1")
    print(tokenizer([
        "Hello, how are you doing?",
        "I'm doing great! How about you?",
        "Good - just chillin"
    ]))
    # Case 3: List of lists representing batches (of multiple conversations)
    # NOTE: Not sure what the max. batch length can be.
    print("Case 3.1")
    print(tokenizer([
        [
            "Hello, how are you doing?",
            "I'm doing great! How about you?",
            "Good - just chillin"
        ],
        [
            "This is speaker 1",
            "This is speaker 2"
        ]
    ]))
    # -- Print the ids of tokens
    print("tokens to ids")
    # The __call__ method generates the input_ids, attention, and speaker_ids.
    print(tokenizer("hello"))
    # The convert ids to tokens take a token (something in the vocab.) and returns
    # its corresponding id (or position) in the vocab. This needs individual tokens.
    print(tokenizer.convert_tokens_to_ids(["hello"]))
    # The encode method converts the text into individual tokens and then returns
    # the ids.
    print(tokenizer.encode("hello"))

def test_spoken_dialogue_tokenizer():
    tokenizer = SpokenDialogueTokenizer("gpt2")
    msg = [
        "Hello, how are you doing?",
        "I'm doing great! How about you?",
        "Good - just chillin",
        "That's good",
        "Just chillin",
        "Good"
    ]
    toks = tokenizer(msg)
    print(toks)
    sp1_idx = np.where(np.asarray(toks['speaker_ids']) == 50259)
    print(sp1_idx)
    sp1_input_ids = np.take(toks['input_ids'], sp1_idx)
    print(sp1_input_ids)
    print(tokenizer.decode(*sp1_input_ids))
    print(tokenizer.sp1_token_id)
    print(tokenizer.sp2_token_id)


def test_huggingface_tokenizer():
    print("Huggingface GPT")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    toks = tokenizer(
        ["Hello, how are you doing?"],return_tensors="pt")
    print("Tokenizer input_ids shape ",toks['input_ids'].shape)
    print("Tokenizer input ids ", toks['input_ids'])
    print(tokenizer.decode(toks['input_ids'][0]))
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    transformer_outputs = model.__call__(**toks)
    print("Model results shape "  ,transformer_outputs[0].shape)


def test_turngpt_initialize():
    turngpt = TurnGPT()
    turngpt.load(
        pretrained_model_name_or_path="gpt2",
        # pretrained_model_name_or_path="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/test_models/epoch=7-step=1128.ckpt",
        model_head="LMHead"
    )
    print(turngpt)



def test_finetune_turngpt():
    pass