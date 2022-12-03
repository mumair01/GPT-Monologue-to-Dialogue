# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:30:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-02 02:58:41

import pytest

import sys
import torch
import os

from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from tests.utils import load_configs, load_text

from typing import List


INFERENCE_TEXT_PATH = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/tests/data/large_text.txt"


def load_inference_text_from_file() -> List:
    """
    Load text data from a text file and separate the paragraphs into lists of
    strings.
    """
    with open(INFERENCE_TEXT_PATH, 'r') as f:
        data = f.read().split("\n\n")
    data = [item.replace("\n", "") for item in data]
    return data


@pytest.fixture
def configs():
    return load_configs()

@pytest.mark.parametrize("model_class", [
   MonologueGPT,
   TurnGPT
])
def test_initialize_conditional_prob_pipe(model_class):
    """Initialize the pipe with the given model"""
    model = model_class()
    model.load()
    pipe = ConditionalProbabilityPipeline(
        model=model,
        N=-1,
        context_buffer_size=512
    )
    assert type(pipe) == ConditionalProbabilityPipeline


@pytest.mark.parametrize("model_class, string_list", [
    # Case 1: Monologue gpt different speakers
    (MonologueGPT, ["<START>", "<SP1>  i haven't seen the keys anywhere  <SP1>",
     "<SP2> have you <SP2>", "<END>"]),
    # Case 2: Monologue gpt same speaker
    (MonologueGPT, [
     "<START>", "<SP1> i haven't seen the keys anywhere have you <SP1>", "<END>"]),
    # Case 3: TurnGPT different speakers
    (TurnGPT, ["sage told me you're going skiing over break", "go on"]),
    # Case 4: TurnGPT same speaker
    (TurnGPT, ["sage told me you're going skiing over break go on"]),
    # NOTE: The tests below may take a long time to run.
    # (MonologueGPT, load_inference_text_from_file()),
    # (TurnGPT, load_inference_text_from_file())
])
def test_conditional_prob_pipe_call(model_class, string_list, configs):
    """
    Use the given model to generate the probabilities for the given string
    using the conditional pipe.
    """
    model = model_class()
    if model_class == MonologueGPT:
        model.load(**configs["monologue_gpt"]["load"])
    elif model_class == TurnGPT:
        model.load(**configs["turngpt"]["load"])
    else:
        raise NotImplementedError(
            f"Model class not implemented: {model_class}"
        )

    pipe = ConditionalProbabilityPipeline(
        model=model,
        **configs["conditional_probability_pipe"]
    )

    probs = pipe(string_list)
    for prob in probs:
        print(prob)
