# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:48:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-10 15:59:16


import pytest
from gpt_dialogue.monologue_gpt import MonologueGPT
from tests.utils import load_configs

@pytest.fixture
def configs():
    return load_configs()

def test_load_default():
    """Load the model with default values """
    model = MonologueGPT()
    model.load()

def test_load_custom(configs):
    """Load the model with the specified arguments"""
    model = MonologueGPT()
    model.load(
        **configs["monologue_gpt"]["load"]
    )


'''
NOTE: By default, as this test will show, GPT-2 does NOT add bos or eos tokens.
It also looks like the GPT-2 tokenizer can only handle strings
or, if it is given
'''
@pytest.mark.parametrize("encodable", [
    # Simple string
    "Hello, how are you doing",
    # List of strings
    [
        "Hello, how are you doing?",
        "I'm doing good! How was dinner?",
        "It was good!"
    ],
    # Lists of lists of strings
    [
        ["Hello, how are you doing?"],
        ["I'm doing good! How was dinner?"],
        ["It was good!"]
    ]

])
def test_encode_decode(encodable, configs):
    """Given a string, encodes and decodes the model."""
    model = MonologueGPT()
    model.load(**configs["monologue_gpt"]["load"])
    encoded = model(encodable)
    decoded = model.tokenizer.decode(encoded["input_ids"])
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
