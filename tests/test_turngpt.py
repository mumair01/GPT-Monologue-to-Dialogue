# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-31 15:39:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-15 12:37:35

import pytest
import sys

from transformers import AutoTokenizer, GPT2LMHeadModel
import transformers
import pytorch_lightning as pl
import numpy as np

from gpt_dialogue.turngpt.tokenizer import SpokenNormalizer, SpokenDialogueTokenizer
from gpt_dialogue.turngpt import TurnGPT

from tests.utils import load_configs


@pytest.fixture
def configs():
    return load_configs()


@pytest.mark.parametrize("add_whitespace_func", [
    True,
    False,
])
def test_spoken_normalizer(add_whitespace_func):
    """Tokenize a single string with the tokenizer"""
    normalizer = SpokenNormalizer()
    string = "Hello,how are you?"
    print(
        f"normalize_str add_whitespace_punc={add_whitespace_func} : {normalizer.normalize_str(string,add_whitespace_punc=add_whitespace_func)}")


@pytest.mark.parametrize("string", [
    # Case 1: Simple string case
    "Hello, how are you doing?",
    # Case 2: List of strings representing multiple turns in a single conversation.
    [
        "Hello, how are you doing?",
        "I'm doing great! How about you?",
        "Good - just chillin"
    ],
    [
        "Hello, how are you doing?",
        "I'm doing great! How about you?",
        "Good - just chillin",
        "That's good",
        "Just chillin",
        "Good"
    ],
    # Case 3: # Case 3: List of lists representing batches (of multiple conversations)
    # NOTE: Not sure what the max. batch length can be.
    [
        [
            "Hello, how are you doing?",
            "I'm doing great! How about you?",
            "Good - just chillin"
        ],
        [
            "This is speaker 1",
            "This is speaker 2"
        ]
    ],
    # Case 5: A conversation with two turns by speaker 1 followed by a turn
    # by speaker 1.
])
def test_spoken_dialogue_tokenizer_call(string, configs):
    """Tokenize the input strings using __call__ method of the tokenizer"""
    tokenizer = SpokenDialogueTokenizer(
        **configs["spoken_dialogue_tokenizer"]
    )
    toks = tokenizer(string, return_tensors="pt")
    print(f"Input string:\n{string}")
    print(f"Result from __call__ method:\n {toks}")


# NOTE: This test shows that the speaker states are split by <ts> tokens.
# An effect of syntactic incoherence might be there since that is the main
# thing the model is learning during pre-training.
@pytest.mark.parametrize("string", [
    # Case 1: A single string with multiple individual turns.
    "i tripped in front of my boss at work today<ts> don't laugh",
    # Case 2: A conversation with two turns by speaker 1 followed by a turn
    # by speaker 1.
    [
        "i tripped in front of my boss at work today<ts> don't laugh",
        "that is really funny"
    ],
    # Case 3: Multiple conversations
    [
        [
            "i tripped in front of my boss at work today<ts> don't laugh",
            "that is really funny",
            "i don't think that was funny<ts> did you",
            "hello one two three"
        ],
        [
            "the cat is on the<ts> the mat",
            "no it is not"
        ]
    ]
])
@pytest.mark.parametrize("split_speaker_by_inline_eos", [
    True, False
])
def test_spoken_dialogue_tokenizer_encode_decode_multi_turn_speaker(
    string, split_speaker_by_inline_eos, configs
):
    """
    In this test, we encode and decode strings where a speaker may have multiple
    turns to confirm that the correct speaker identities are assigned per turn.
    """
    tokenizer = SpokenDialogueTokenizer(
        **configs["spoken_dialogue_tokenizer"]
    )
    toks = tokenizer(
        string,
        split_speaker_by_inline_eos=split_speaker_by_inline_eos
    )
    print(f"Input string: {string}")
    print(f"split_speaker_by_inline_eos: {split_speaker_by_inline_eos}")
    for speaker in (1, 2):
        sp_decoded = tokenizer.decode_speaker(toks, speaker)
        print(f"Turns by speaker {speaker}")
        print(sp_decoded)


@pytest.mark.parametrize("string", [
    # Simple string
    "hello",
])
def test_spoken_dialogue_tokenizer_encode(string, configs):
    """Tokenize the input strings using the encode method of the tokenizer"""
    tokenizer = SpokenDialogueTokenizer(
        **configs["spoken_dialogue_tokenizer"]
    )
    print(f"Input string:\n{string}")
    print(f"Result from encode method: {tokenizer.encode(string)}")


@pytest.mark.parametrize("string", [
    # Simple string
    "hello",
])
def test_spoken_dialogue_tokenizer_tokens_to_ids(string, configs):
    """
    Use the convert_tokens_to_ids method and compare with input_ids from
    encode method.
    """
    tokenizer = SpokenDialogueTokenizer(
        **configs["spoken_dialogue_tokenizer"]
    )
    if type(string) == str:
        toks = tokenizer.convert_tokens_to_ids([string])
    else:
        toks = tokenizer.convert_tokens_to_ids(string)
    assert toks == tokenizer(string)["input_ids"]


@pytest.mark.parametrize("string, speaker", [
    # Simple string
    ([
        "Hello, how are you doing?",
        "I'm doing great! How about you?",
        "Good - just chillin"
    ], 1),
    ([
        "Hello, how are you doing?",
        "I'm doing great! How about you?",
        "Good - just chillin"
    ], 2)
])
def test_spoken_dialogue_tokenizer_decode_speaker_ids(string, speaker, configs):
    """
    Use the tokenizer __call__ method on string and decode output from a specific
    speaker only.
    """
    tokenizer = SpokenDialogueTokenizer(
        **configs["spoken_dialogue_tokenizer"]
    )
    toks = tokenizer(string)
    decoded = tokenizer.decode_speaker(toks, speaker)
    print(f"Input string: {string}")
    print(f"Decoded string for {speaker}: {decoded}")


def test_load_default():
    """Load the model with default values """
    model = TurnGPT()
    model.load()


def test_load_custom(configs):
    """Load the model with the specified arguments"""
    model = TurnGPT()
    model.load(
        **configs["turngpt"]["load"]
    )
