# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-09-23 15:30:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-09-23 15:32:46

import pytest

import sys
import os

from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline


def test_pipe_monologue_gpt():

    mono_model = MonologueGPT()
    mono_model.load()
    mono_tokenizer = mono_model.tokenizer

    toks = mono_tokenizer(
        "sage told me you're going skiing over break go on"
    )
    print(mono_tokenizer.decode(toks["input_ids"]))

    mono_model.model.eval()
    pipe = ConditionalProbabilityPipeline(
        model=mono_model,
        N=-1,
        context_buffer_size=512
    )
    probs = pipe(["sage told me you're going skiing over break go on"])
    for prob in probs:
        print(prob)


def test_pipe_turngpt():
    turngpt = TurnGPT()
    turngpt.load(
        pretrained_model_name_or_path="gpt2",
        model_head="DoubleHeads"
    )
    turngpt_tokenizer = turngpt.tokenizer

    toks = turngpt_tokenizer(
        "sage told me you're going skiing over break go on"
    )
    print(toks)
    print(turngpt_tokenizer.decode(toks["input_ids"]))

    pipe = ConditionalProbabilityPipeline(
        model=turngpt,
        N=-1,
        context_buffer_size=512
    )
    probs = pipe(["sage told me you're going skiing over break go on"])
    for prob in probs:
        print(prob)