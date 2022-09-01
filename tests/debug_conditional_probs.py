# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-30 13:07:33
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-31 13:07:26

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import transformers

from gpt_dialogue.monologue_gpt import MonologueGPT
from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.pipelines import ConditionalProbabilityPipeline


if __name__ == "__main__":

    print("Monologue GPT")
    mono_model = MonologueGPT()
    mono_model.load()
    mono_tokenizer = mono_model.tokenizer
    toks = mono_tokenizer(
        "sage told"
    )
    # print(toks)
    # print(mono_tokenizer("sage"))
    # print(mono_tokenizer("told"))
    # print(mono_tokenizer.decode(44040))
    # print(mono_tokenizer.decode(1297))

    # print(mono_tokenizer.decode(toks["input_ids"]))

    mono_model.model.eval()
    pipe = ConditionalProbabilityPipeline(
        model=mono_model,
        N=-1,
        context_buffer_size=512
    )
    probs = pipe(["sage told me you're going skiing over break go on"])
    for prob in probs:
        print(prob)

    print("TurnGPT")
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

