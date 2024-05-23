# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:48:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 13:24:54


import pytest
from LLMDialogue.models import GPT2


def test_load_default():
    """Load the model with default values"""
    model = GPT2()
    model.load()


def test_load_custom(gpt2_load_configs):
    """Load the model with the specified arguments"""
    model = GPT2()
    model.load()


"""
NOTE: By default, as this test will show, GPT-2 does NOT add bos or eos tokens.
It also looks like the GPT-2 tokenizer can only handle strings
or, if it is given
"""


@pytest.mark.parametrize(
    "encodable",
    [
        # Simple string
        "Hello, how are you doing",
        # List of strings
        [
            "Hello, how are you doing?",
            "I'm doing good! How was dinner?",
            "It was good!",
        ],
        # Lists of lists of strings
        [
            ["Hello, how are you doing?"],
            ["I'm doing good! How was dinner?"],
            ["It was good!"],
        ],
    ],
)
def test_encode_decode(encodable, gpt2_model):
    """Given a string, encodes and decodes the model."""
    model = gpt2_model
    encoded = model.encode(encodable)
    decoded = model.tokenizer.decode(encoded["input_ids"])
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
