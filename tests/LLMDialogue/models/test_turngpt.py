# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 15:51:40
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-19 19:00:24
import pytest
import sys

from transformers import AutoTokenizer, GPT2LMHeadModel
import transformers
import pytorch_lightning as pl
import numpy as np


from LLMDialogue.models import TurnGPT


def test_load_default():
    """Load the model with default values"""
    model = TurnGPT()
    model.load()


def test_load_custom(turngpt_load_configs):
    """Load the model with the specified arguments"""
    model = TurnGPT()
    model.load(**turngpt_load_configs)
