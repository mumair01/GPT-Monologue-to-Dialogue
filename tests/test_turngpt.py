# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-31 15:39:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-03 13:21:54

import pytest
import sys

from transformers import AutoTokenizer, GPT2LMHeadModel
import transformers


from gpt_dialogue.turngpt.tokenizer import SpokenNormalizer, SpokenDialogueTokenizer
from gpt_dialogue.turngpt.model import TurnGPT


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


def test_model():
    # Loading model with all default configs
    model = TurnGPT(
        pretrained_model_name_or_path="gpt2",
        load_pretrained_configs=True,
        learning_rate= 1e-4)
    tokenizer = SpokenDialogueTokenizer("gpt2")

    # print(model.tokenize([
    #     [
    #         "Hello, how are you doing?",
    #         "I'm doing great! How about you?",
    #         "Good - just chillin"
    #     ],
    #     [
    #         "Hello, how are you doing?",
    #         "Good - just chillin"
    #     ],

    # ]))
    # print(model.tokenize([
    #     "Hello, how are you doing?",
    #     "I'm doing great! How about you?",
    #     "Good - just chillin"
    # ]))
    # print(tokenizer.pad_token_id, tokenizer.pad_token)
    # print(model.tokenize("Hello"))

    # Tokenize and run through model.
    tokens = tokenizer(["Hello, how are you doing?"],return_tensors="pt",)
    print(tokens['input_ids'].shape)
    print(tokens)
    print(model._transformer.config)
    print(model(**tokens))


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

    print("TurnGPT")
    turngpt = TurnGPT(
        pretrained_model_name_or_path="gpt2",
        load_pretrained_configs=True,
        learning_rate= 1e-4)
    toks = turngpt.tokenize(["Hello, how are you doing?"],return_tensors="pt")
    print(toks['input_ids'].shape)
    print(toks['input_ids'])
    print(turngpt.decode(toks['input_ids'][0]))
    out =  turngpt(**toks)
    print(out['logits'].shape)


def test_original():
    model = TurnGPT()
    model.init_tokenizer()
    model.initialize_special_embeddings()
    toks = model.tokenize_strings(["Hello, how are you doing?"])
    print(toks)
    print(toks['input_ids'].shape)
    print(model.tokenizer.decode(toks['input_ids'][0]))
    out = model(**toks)

