# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-11-27 15:42:39
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-15 15:56:21

import pytest
import sys

import transformers
import pytorch_lightning as pl
import numpy as np
import torch

from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    "string",
    [
        "Hello",
        # List of strings
        # ["Hello, how are you doing?"]
    ],
)
def test_huggingface_tokenizer(string):
    """Various test for transformers"""
    print("Huggingface GPT")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    toks = tokenizer(string, return_tensors="pt")
    print(toks)
    print("Tokenizer input_ids shape ", toks["input_ids"].shape)
    print("Tokenizer input ids ", toks["input_ids"])
    print(tokenizer.decode(toks["input_ids"][0]))
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    transformer_outputs = model.__call__(**toks)
    print("Model results shape ", transformer_outputs[0].shape)


def test_gpt():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(model)
    print(tokenizer)

    context = "i haven't seen the keys"
    turns_so_far = "i haven't seen the keys anywhere"

    context_encoding = tokenizer(context, return_tensors="pt")
    whole_text_encoding = tokenizer(turns_so_far, return_tensors="pt")

    cw_encoding = {
        k: v[:, context_encoding["input_ids"].shape[1] :]
        for k, v in whole_text_encoding.items()
    }

    print("Context ", tokenizer.decode(*context_encoding["input_ids"]))
    print("cw ", tokenizer.decode(*cw_encoding["input_ids"]))

    whole_text_encoding_shape = whole_text_encoding["input_ids"].shape[1]
    context_encoding_shape = context_encoding["input_ids"].shape[1]

    with torch.no_grad():
        output = model(**whole_text_encoding)

    cw_extracted_logits = output.logits[
        -1, context_encoding["input_ids"].shape[1] - 1 : -1, :
    ]

    softmax = torch.nn.Softmax(dim=-1)
    cw_extracted_probs_from_logits = softmax(cw_extracted_logits)

    print("cw_extracted_probs_from_logits ", cw_extracted_probs_from_logits)

    cw_extracted_log_probs_from_logits = torch.log(
        cw_extracted_probs_from_logits
    )

    print(
        "cw_extracted_log_probs_from_logits ",
        cw_extracted_log_probs_from_logits,
    )

    cw_tokens_probs = []
    for cw_subtoken, probs in zip(
        cw_encoding["input_ids"][0], cw_extracted_log_probs_from_logits
    ):
        cw_tokens_probs.append(probs[cw_subtoken])

    print("cw_tokens_probs ", cw_tokens_probs)
    prob = float(torch.exp(torch.sum(torch.Tensor(cw_tokens_probs))))

    print(prob)
