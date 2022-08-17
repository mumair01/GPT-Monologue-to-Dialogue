# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 16:16:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-12 17:13:53


import pytest
from gpt_dialogue.turngpt.turngpt import TurnGPT
from gpt_dialogue.pipelines.conditional_probs import ConditionalProbabilityPipeline

import pprint

def test():

    model = TurnGPT()
    model.load()

    pipe = ConditionalProbabilityPipeline(
        model=model,
        N=-1,
        context_buffer_size = 512,
        normalize=False
    )
    res = pipe(
        utterances=[
            "Hello",
            "how are you doing?"
        ]
    )
    pprint.pprint(res)