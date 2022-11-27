# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:48:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-10-07 14:46:05


import pytest

from gpt_dialogue.monologue_gpt import MonologueGPT

_TOKENIZER_EOS_TOKEN = "<|endoftext|>"
_TOKENIZER_PAD_TOKEN = "<PAD>"
_TOKENIZER_ADDITIONAL_SPECIAL_TOKENS = [
    "<SP1>", # Speaker 1 token
    "<SP2>", # Speaker 2 token
    "<START>", # Conversation start token
    "<END>" # Conversation end token
]

def test_initialize():
    model = MonologueGPT()
    model.load(
        model_checkpoint="gpt2"
    )

def test_initialize_additional_tokens():
    model = MonologueGPT()
    model.load(
        model_checkpoint="gpt2",
        tokenizer_pad_token=_TOKENIZER_PAD_TOKEN,
        tokenizer_eos_token=_TOKENIZER_EOS_TOKEN,
        tokenizer_additional_special_tokens=_TOKENIZER_ADDITIONAL_SPECIAL_TOKENS
    )

def test_finetune_monologue_gpt():
    pass