# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:48:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-11-27 16:48:09


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
