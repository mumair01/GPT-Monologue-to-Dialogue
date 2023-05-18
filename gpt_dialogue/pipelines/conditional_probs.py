# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-12 15:30:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-18 08:55:37

"""
Contains the implementation of the conditional probability pipeline that 
can be used to make inferences given a model and inference data
"""

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
from collections import defaultdict
from dataclasses import dataclass

from gpt_dialogue.model import LanguageModel

import logging

logger = logging.getLogger(__name__)


@dataclass
class Param:
    value: Any = None
    required: bool = True


class CPPipeOutput:
    """
    Wrapper on top of the ConditionalProbabilityPipeline output which
    provided convenience methods for interacting with the output.
    """

    def __init__(self, pipe_output: Dict):
        """
        Parameters
        ----------
        pipe_output : Dict
            Output of the conditional probability pipeline
        """
        self.pipe_output = pipe_output
        self.turns = self.collect_turns()

    def __iter__(self):
        """Internal iterator for each element in the output"""
        for item in self.pipe_output:
            yield item

    def raw_output(self) -> Dict:
        """
        Obtain the complete underlying output from the ConditionalProbabilityPipeline

        Returns
        -------
        Dict
           Complete ConditionalProbabilityPipeline output
        """
        return self.pipe_output

    def number_of_turns(self) -> int:
        """
        Obtain the number of turns in the underlying output

        Returns
        -------
        int
            Number of turns in the output
        """
        return len(self.turns)

    def collect_turns(self) -> Dict:
        """
        Given the ConditionalProbabilityPipe output, separates the output based
        on turns.
        """
        data = defaultdict(lambda: list())
        for item in self.pipe_output:
            data[item["turn_no"]].append(
                {k: v for k, v in item.items() if k != "turn_no"}
            )
        return data

    def probability_of_turn_no(self, turn_no: int) -> torch.Tensor:
        """
        Assuming that pipe_output contains the output of the ConditionalProbabilityPipe
        with N = -1, obtain the word probabilities for each word in the turn.

        Parameters
        ----------
        turn_no : int
            Turn number, starting from 0

        Returns
        -------
        torch.Tensor
            Tensor containing word probabilities for each word in the specified
            turn.
        """
        return torch.tensor(
            [item["last_word_prob"] for item in self.turns[turn_no]]
        )

    def turn_text(self, turn_no: int) -> str:
        """
        "Get the complete text of the given turn"

        Parameters
        ----------
        turn_no : int
            Turn number, starting from 0

        Returns
        -------
        str
            Text of the specified turn
        """
        return " ".join([item["word"] for item in self.turns[turn_no]])

    def word_probabilities_of_matched_string(self, match_string: str) -> float:
        """
        Return the probabilities of words in a turn if it matches the given string

        Parameters
        ----------
        match_string : str
            The text of a potential turn in the output.

        Returns
        -------
        float
            The probability of each word in a turn in the output if it matches the
            given match_string

        Raises
        ------
        Exception
            Raised if the specified match_string is not found in the output
        """
        # Collect all the turns and the corresponding text.
        turns_text = {
            item["turn_no"]: self.turn_text(item["turn_no"])
            for item in self.pipe_output
        }
        for turn_no, text in turns_text.items():
            if text == match_string:
                return self.probability_of_turn_no(turn_no)
        raise Exception(
            f"ERROR: No turn found with the given string: {match_string}\n{turns_text}"
        )

    def probability_of_matched_string(self, match_string: str) -> float:
        """
        Obtain the total probability of a given match_string

        Parameters
        ----------
        match_string : str
            The text of a potential turn in the output.

        Returns
        -------
        float
            Total output probability of the provided turn.
        """
        word_probs = self.word_probabilities_of_matched_string(match_string)
        return self._multiply_probabilities(word_probs)

    def _multiply_probabilities(self, probs) -> float:
        """
        Generate the product of all the probabilities in a given list.
        """
        return float(torch.exp(torch.sum(torch.log(torch.tensor(probs)))))


# Pipeline that loads a causal language Model with a __call__ method and
#     tokenizer and uses it to generate conditional probabilities.
#     Each __call__ assumes that it is processing a single `conversation` -
#     represented by a list of utterances at a time.


class ConditionalProbabilityPipeline:
    """
    Pipeline that consumes a given causal language model i.e, model that
    generates P(word | context), and uses it to generate conditional probabilities
    for each word in a given input, conditioned on the previous context.
    """

    # Parameters that are required by the pipeline
    _PARAMS = {
        "model": Param(required=True),
        "N": Param(required=True),
        "context_buffer_size": Param(required=True),
    }

    def __init__(self, **kwargs):
        """
        Initialize the pipeline using kwargs, which must follow the template
        defined by _PARAMS
        """
        # Sanitize all parameters
        self._sanitize_parameters(**kwargs)

        # Initialize torch device
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = self._PARAMS["model"].value
        logger.info(
            f"Pipe {self} initialized with device: {self.device.type}, "
            f"using model: {model}"
        )
        if self.device.type == "cpu":
            logger.warning(
                f"WARNING: Using device {self.device.type} in pipe "
                "is slow - switch to gpu if possible"
            )

    def __call__(self, utterances: List[str]) -> CPPipeOutput:
        """
        Generate P(word | context) for each word in the given utterances.

        Parameters
        ----------
        utterances : List[str]
           Represents a number of turns that are part of the same conversation.
           Each item of the list is a new turn.

        Returns
        -------
        CPPipeOutput
            Output object with convenience methods
        """
        return CPPipeOutput(
            self._postprocess(self._forward(self._preprocess(utterances)))
        )

    def _preprocess(self, utterances: List[str]):
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
            context_buffer_size=context_buffer_size,
        )

    def _postprocess(self, forward_outputs):
        """Prepare _forward outputs to be returned."""
        return forward_outputs

    ############################## INFERENCE METHODS ###########################

    def _generate_conditional_probabilities(
        self, model, utterances, N, context_buffer_size
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

                last_word_prob = self._get_last_word_prob(
                    model=model, text_so_far=text_so_far, context=context
                )

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

    def _get_last_word_prob(self, model, text_so_far, context):
        # NOTE: We need to make sure there are no empty strings that go to the
        # tokenizer.
        # print("text so far: ",text_so_far)
        turns_so_far = [
            " ".join(turn_words).strip() for turn_words in text_so_far
        ]
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

        whole_text_encoding = whole_text_encoding.to(self.device)
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
        cw_extracted_log_probs_from_logits = torch.log(
            cw_extracted_probs_from_logits
        )

        # Extract the probabilities of the specific tokens
        cw_tokens_probs = []
        for cw_subtoken, probs in zip(
            cw_encoding["input_ids"][0], cw_extracted_log_probs_from_logits
        ):
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
        self, text_so_far: List[List[str]], context_buffer_size
    ):
        while True:
            num_words_so_far = len(list(itertools.chain(*text_so_far)))
            if num_words_so_far < context_buffer_size:
                break
            while (
                len(list(itertools.chain(*text_so_far))) >= context_buffer_size
            ):
                text_so_far[0].pop(0)
        return text_so_far
