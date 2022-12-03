# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 15:30:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-01 03:28:16

from distutils import text_file
from email import message
import sys
import os
from typing import List, Any, Dict
import itertools

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

from copy import deepcopy
from dataclasses import dataclass

from gpt_dialogue.model import LanguageModel

import logging
logger = logging.getLogger(__name__)


@dataclass
class Param:
    value: Any = None
    required: bool = True


class ConditionalProbabilityPipeline:
    """
    Pipeline that loads a causal language Model with a __call__ method and
    tokenizer and uses it to generate conditional probabilities.
    Each __call__ assumes that it is processing a single `conversation` -
    represented by a list of utterances at a time.
    """

    _PARAMS = {
        "model": Param(required=True),
        "N": Param(required=True),
        "context_buffer_size": Param(required=True)
    }

    def __init__(self, **kwargs):

        # Sanitize all parameters
        self._sanitize_parameters(**kwargs)

        # Initialize torch device
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Pipe {self} initialized with device: {self.device.type}")
        if self.device.type == "cpu":
            logger.warning(f"WARNING: Using device {self.device.type} in pipe "
                           "is slow - switch to gpu if possible")

    def __call__(self, utterances: List[str]):
        """Processes a list of strings containing utterances"""
        return self.postprocess(self._forward(self.preprocess(utterances)))

    def preprocess(self, utterances: List[str]):
        """Prepare inputs given to __call__ for _forward"""
        return {"utterances": utterances}

    def _forward(self, preprocess_output: Dict):

        # Get the required param values
        N = self._PARAMS["N"].value
        model: LanguageModel = self._PARAMS["model"].value
        context_buffer_size = self._PARAMS["context_buffer_size"].value
        utterances = preprocess_output["utterances"]

        # Move the model to GPU before running.
        model.to(self.device)
        # Put the model in eval mode
        model.eval()

        # Generate the probabilities and return the list
        return self._generate_conditional_probabilities(
            model=model,
            utterances=utterances,
            N=N,
            context_buffer_size=context_buffer_size
        )

    def postprocess(self, forward_outputs):
        """Prepare _forward outputs to be returned. """
        return forward_outputs

    ############################## INFERENCE METHODS ###########################

    def _generate_conditional_probabilities(
        self,
        model,
        utterances,
        N,
        context_buffer_size
    ):
        # We need to maintain a whole text list
        text_so_far: List[List[str]] = []
        results = []

        pbar = tqdm(desc="Processing turns", total=len(utterances))

        for turn_no, turn in enumerate(utterances):
            split_turn = turn.strip().split()
            turn_length = len(split_turn)
            n_probs = turn_length if N == -1 or N > turn_length else N

            current_turn_words = []
            text_so_far.append([])

            for word_no in range(turn_length - n_probs, turn_length):
                current_turn_words = split_turn[:word_no+1]
                # Update the text lists
                text_so_far[turn_no] = current_turn_words

                # Trim the text so far if it is exceeding buffer length
                text_so_far = self._trim_to_context_buffer_size(
                    text_so_far, context_buffer_size)

                # Context is the sentence so far + everything upto the last word
                context = text_so_far[:turn_no] + [split_turn[:word_no]]
                context = self._trim_to_context_buffer_size(
                    context, context_buffer_size
                )

                # By def., conditional probability does not occur for the first word.
                if turn_no == 0 and word_no == 0:
                    continue

                last_word_prob = self._get_last_word_prob(
                    model=model,
                    text_so_far=text_so_far,
                    context=context
                )

                # Adding to results
                results.append({
                    "turn_no": turn_no,
                    "word_no": word_no,
                    "context": " ".join([" ".join(turn_words) for turn_words in context]),
                    "word": current_turn_words[-1],
                    "last_word_prob": last_word_prob
                })
            pbar.update()

        return results

    def _get_last_word_prob(
        self,
        model,
        text_so_far,
        context
    ):

        # NOTE: We need to make sure there are no empty strings that go to the
        # tokenizer.
        turns_so_far = [" ".join(turn_words).strip()
                        for turn_words in text_so_far]
        turns_so_far = [item for item in turns_so_far if len(item) > 0]
        context = [" ".join(turn_words).strip() for turn_words in context]
        context = [item for item in context if len(item) > 0]

        context_encoding = model.encode(
            context, return_tensors="pt"
        )
        whole_text_encoding = model.encode(
            turns_so_far, return_tensors="pt"
        )
        cw_encoding = {
            k: v[:, context_encoding["input_ids"].shape[1]:]
            for k, v in whole_text_encoding.items()
        }
        # NOTE: We assume that each additional work MUST add tokens to the encoding.
        whole_text_encoding_shape = whole_text_encoding["input_ids"].shape[1]
        context_encoding_shape = context_encoding["input_ids"].shape[1]
        assert whole_text_encoding_shape > context_encoding_shape, \
            (f"Dims mismatch, whole encoding {whole_text_encoding_shape} must"
             f" be greater than context encoding {context_encoding_shape}")

        whole_text_encoding = whole_text_encoding.to(self.device)
        output = model(**whole_text_encoding)
        # Obtain the logits for the last hidden state and the logits
        # that provide values for the tokens in the critical word.
        # i.e., if cw token starts at position i in the sentence, then the logits
        # are from i-1 to len(tokens) - 1.
        cw_extracted_logits = output.logits[-1,
                                            context_encoding["input_ids"].shape[1]-1:-1, :]
        # Obtain the probabilities from the logits
        softmax = torch.nn.Softmax(dim=-1)
        cw_extracted_probs_from_logits = softmax(cw_extracted_logits)

        # NOTE: Converting to log scale and taking exponential sum of the log
        # probabilities at the end will ensure that there is not floating point
        # overflow issue for very small probability values.
        cw_extracted_log_probs_from_logits = torch.log(
            cw_extracted_probs_from_logits)

        # Extract the probabilities of the specific tokens
        cw_tokens_probs = []
        for cw_subtoken, probs in zip(cw_encoding["input_ids"][0], cw_extracted_log_probs_from_logits):
            cw_tokens_probs.append(probs[cw_subtoken])
        return float(torch.exp(torch.sum(torch.Tensor(cw_tokens_probs))))

    ################################# HELPER METHODS ###########################

    def _sanitize_parameters(self, **kwargs):
        """Clean up the input kwargs and assign to the appropriate map."""
        for param_dict in (self._PARAMS,):
            for k, v in param_dict.items():
                # Required args must be present.
                if v.required and not k in kwargs:
                    raise Exception(
                        f"ERROR: Value not specified for required parameter {k}"
                    )
                if k in kwargs:
                    v.value = kwargs[k]

    def _trim_to_context_buffer_size(
        self,
        text_so_far: List[List[str]],
        context_buffer_size
    ):
        while True:
            num_words_so_far = len(list(itertools.chain(*text_so_far)))
            if num_words_so_far < context_buffer_size:
                break
            while len(list(itertools.chain(*text_so_far))) >= context_buffer_size:
                text_so_far[0].pop(0)
        return text_so_far
