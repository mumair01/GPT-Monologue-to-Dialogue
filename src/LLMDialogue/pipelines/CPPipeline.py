# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-22 08:12:56
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-31 13:10:25


import sys
import os
import numpy as np
import torch
from torch.nn.functional import softmax
import hashlib
import pytorch_lightning as pl
import itertools
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Any, Callable, Optional, Tuple

from LLMDialogue.baseConfigs import BaseConfigs
from LLMDialogue.models import LanguageModel, Model
from LLMDialogue.datasets import DatasetBuilder, DatasetConfig, load
from LLMDialogue.pipelines.base import PipelineBuilder, PipelineConfig
from LLMDialogue.utils.general import (
    get_dirname,
    remove_dir,
    write_json,
    remove_file,
    make_tarfile,
    get_files_in_dir,
    get_subdirs_in_dir,
    read_json,
    get_system_device,
)
from LLMDialogue.utils.logging import log_format

import logging

logger = logging.getLogger(__name__)


_PIPELINE_NAME = "ConditionalProbabilityPipeline"

_PROBABILITY_METHODS = (
    "words",
    "eos",
)


class CPPipelineConfig(PipelineConfig):

    def __init__(
        self,
        model: LanguageModel,
        utterances: List[str],
        N: int,
        context_buffer_size: int,
        probability_method: str,
        **kwargs,
    ):
        assert (
            probability_method in _PROBABILITY_METHODS
        ), f"Method must be in {_PROBABILITY_METHODS}"
        config_kwargs = {
            "name": _PIPELINE_NAME,
            "model": model,
            "utterances": utterances,
            "N": N,
            "context_buffer_size": context_buffer_size,
            "probability_method": probability_method,
        }

        # Remove a key if it exists in kwargs.
        config_keys = list(config_kwargs.keys())
        for k in config_keys:
            if k in kwargs:
                del config_kwargs[k]
        # Initialize the superclass.
        super().__init__(**kwargs, **config_kwargs)

    def to_dict(self) -> Dict:
        # Here, make sure we remove items that we are not serializable.
        d = self.__dict__
        d["model"] = str(type(d["model"]))
        return d


class CPPipeline(PipelineBuilder):
    def __init__(self, config: Optional[CPPipelineConfig]):
        super().__init__(config)

    def save_to_disk(self) -> None:
        assert os.path.isdir(self.config.archive_dir)
        super()._save_to_disk(deepcopy(self.config), self.data)

    def __str__(self) -> str:
        return self.config.__str__()

    ###########
    # REQUIRED PUBLIC METHODS
    ###########

    def run(self):
        # Save the data attribute
        self.data = self._create_data()
        # Save the dataset to disk.
        self.save_to_disk()

    ###########
    # REQUIRED PRIVATE METHODS
    ###########

    def load_from_disk(self) -> Dict:
        # Load the configs and the actual data form the data file and the
        # info files.
        # TODO: We can't do this yet because we can't serialize the model
        # and the dm to save them.
        data = super()._load_data_from_disk(fname=self._DATA_FNAME)
        configs = super()._load_data_from_disk(fname=self._CONF_FNAME)
        return {"configs": configs, "data": data}

    ###########
    # Pipeline SPECIFIC PRIVATE METHODS
    ###########

    def _create_data(self) -> Dict:
        # Move the model to gpu
        model = self.config.model
        model.to(self.config.device)
        # Put the model in eval mode
        model.eval()

        data = self._generate_conditional_probabilities(
            model,
            self.config.utterances,
            self.config.N,
            self.config.context_buffer_size,
        )
        return data

    def _generate_conditional_probabilities(
        self,
        model,
        utterances,
        N,
        context_buffer_size,
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
                current_turn_words = split_turn[: word_no + 1]
                # Update the text lists
                text_so_far[turn_no] = current_turn_words

                # Trim the text so far if it is exceeding buffer length
                text_so_far = self._trim_to_context_buffer_size(
                    text_so_far, context_buffer_size
                )

                # Context is the sentence so far + everything upto the last word
                context = text_so_far[:turn_no] + [split_turn[:word_no]]
                context = self._trim_to_context_buffer_size(
                    context, context_buffer_size
                )

                # By def., conditional probability does not occur for the first word.
                if turn_no == 0 and word_no == 0:
                    continue

                # USe the correct prob. method
                prob_method = (
                    self._get_last_word_prob_using_words
                    if self.config.probability_method == "words"
                    else self._get_last_word_prob_using_eos
                )
                last_word_prob = prob_method(model, text_so_far, context)

                # Adding to results
                results.append(
                    {
                        "turn_no": turn_no,
                        "word_no": word_no,
                        "context": " ".join(
                            [" ".join(turn_words) for turn_words in context]
                        ),
                        "word": current_turn_words[-1],
                        "last_word_prob": last_word_prob,
                    }
                )
            pbar.update()

        return results

    def _get_last_word_prob_using_eos(
        self,
        model,
        text_so_far: List[str],
        context: List[str],
    ):
        # Make sure there are no empty tokens.
        turns_so_far = [" ".join(turn_words).strip() for turn_words in text_so_far]
        turns_so_far = [item for item in turns_so_far if len(item) > 0]
        context = [" ".join(turn_words).strip() for turn_words in context]
        context = [item for item in context if len(item) > 0]
        # In this strategy, we simply need to get the probability of the EOS
        # token after the last word.
        whole_text_encoding = model.encode(turns_so_far, return_tensors="pt")
        whole_text_decoded = model.decode(*whole_text_encoding["input_ids"])

        # Log this information for debugging later.
        msg = (
            f"Text so far: {text_so_far}\n"
            f"Turns so far: {turns_so_far}\n"
            f"Context: {context}\n"
            f"Whole text encoding: {whole_text_encoding}\n"
            f"whole text decoding: {whole_text_decoded}"
        )
        logging.debug(msg)

        # Now, predict using the model
        whole_text_encoding = whole_text_encoding.to(self.config.device)
        with torch.no_grad():
            output = model(**whole_text_encoding)

        # Get the logits for the distribution after the end of the last word
        # in the whole text encoding.
        seq_len = whole_text_encoding["input_ids"].shape[1]
        last_logits = output["logits"][-1, seq_len - 1]
        # Softmax for the probability distribution.
        last_probs = softmax(last_logits, dim=-1)
        # Get the probability of the EOS token (as known by the tokenizer.)
        # TODO: Potentially add a config argument to specify the token - but
        # this should likely be the same as the one known by the tokenizer.
        eos_prob = last_probs[model.tokenizer.eos_token_id]
        return float(eos_prob)

    def _get_last_word_prob_using_words(
        self,
        model,
        text_so_far,
        context,
    ):
        # NOTE: We need to make sure there are no empty strings that go to the
        # tokenizer.
        # print("text so far: ",text_so_far)
        turns_so_far = [" ".join(turn_words).strip() for turn_words in text_so_far]
        # print("turns so far: ", turns_so_far)
        turns_so_far = [item for item in turns_so_far if len(item) > 0]
        context = [" ".join(turn_words).strip() for turn_words in context]
        context = [item for item in context if len(item) > 0]

        context_encoding = model.encode(context, return_tensors="pt")
        whole_text_encoding = model.encode(turns_so_far, return_tensors="pt")

        cw_encoding = {
            k: v[:, context_encoding["input_ids"].shape[1] :]
            for k, v in whole_text_encoding.items()
        }

        # Logging information useful for debugging.
        cw_decoded = model.decode(*cw_encoding["input_ids"])
        msg = (
            f"Text so far: {text_so_far}\n"
            f"Turns so far: {turns_so_far}\n"
            f"Context: {context}\n"
            f"Whole text encoding: {whole_text_encoding}\n"
            f"cw decoding: {cw_decoded}"
        )
        logging.debug(msg)

        # NOTE: We assume that each additional work MUST add tokens to the encoding.
        whole_text_encoding_shape = whole_text_encoding["input_ids"].shape[1]
        context_encoding_shape = context_encoding["input_ids"].shape[1]
        assert whole_text_encoding_shape > context_encoding_shape, (
            f"Dims mismatch, whole encoding {whole_text_encoding_shape} must"
            f" be greater than context encoding {context_encoding_shape}"
        )

        whole_text_encoding = whole_text_encoding.to(self.config.device)
        output = model(**whole_text_encoding)
        # Obtain the logits for the last hidden state and the logits
        # that provide values for the tokens in the critical word.
        # i.e., if cw token starts at position i in the sentence, then the logits
        # are from i-1 to len(tokens) - 1.
        cw_extracted_logits = output.logits[
            -1, context_encoding["input_ids"].shape[1] - 1 : -1, :
        ]
        # Obtain the probabilities from the logits
        softmax = torch.nn.Softmax(dim=-1)
        cw_extracted_probs_from_logits = softmax(cw_extracted_logits)

        # NOTE: Converting to log scale and taking exponential sum of the log
        # probabilities at the end will ensure that there is no floating point
        # overflow issue for very small probability values.
        cw_extracted_log_probs_from_logits = torch.log(cw_extracted_probs_from_logits)

        # Extract the probabilities of the specific tokens
        cw_tokens_probs = []
        for cw_subtoken, probs in zip(
            cw_encoding["input_ids"][0], cw_extracted_log_probs_from_logits
        ):
            cw_tokens_probs.append(probs[cw_subtoken])
        return float(torch.exp(torch.sum(torch.Tensor(cw_tokens_probs))))

    def _trim_to_context_buffer_size(
        self, text_so_far: List[List[str]], context_buffer_size
    ):
        while True:
            num_words_so_far = len(list(itertools.chain(*text_so_far)))
            if num_words_so_far < context_buffer_size:
                break
            while len(list(itertools.chain(*text_so_far))) >= context_buffer_size:
                text_so_far[0].pop(0)
        return text_so_far
