# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 10:26:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-01 16:23:44

############################
# This module is a re-implementation of the TurnGPT tokenizer as a comparison to the
# speaker identity embedding approach that we have taken.
# NOTE: This code is taken from the repository specified below and may or may
# not be modified.
# Acknowledgements:
#   Paper: https://arxiv.org/abs/2010.10874
#   Code: https://github.com/ErikEkstedt/TurnGPT
############################

import re
import sys
from typing import List, Union
from collections import defaultdict
from datasets import load_dataset

from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch

from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)


import logging
logger = logging.getLogger(__name__)


TOKENIZER_PAD_TOKEN = "<|endoftext|>"
TOKENIZER_EOS_TOKEN = "<ts>"
BASE_SPEAKER_TOKEN = "<SP{}>"
CONV_START_TOKEN = "<START>"
CONV_END_TOKEN = "<END>"


# TODO: Potentially change the PAD token. Originally, the <ts> token is used
# as the end of text to indicate 'turn-switch', and each word is treated as
# a separate sequence (since there are no turns initially).
class SpokenNormalizer:
    """
    Normalizer to preprocess text (e.g, clear punctuation etc.) before it is
    passed to a tokenizer. This is a wrapper on top of the huggingface Sequence.
    Normalizers Link: https://huggingface.co/docs/tokenizers/api/normalizers#normalizers
    """

    def __init__(self):
        self.normalizer = Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),  # punctuation
                Replace(Regex(r"\s\s+"), " "),  # double spaces
                Strip(),
            ]
        )

    def normalize_str(self,s, add_whitespace_punc=True):
        if add_whitespace_punc:
            s = self.__add_whitespace_after_punctuation(s)
        return self.normalizer.normalize_str(s)

    def __add_whitespace_after_punctuation(self, s):
        s = re.sub(r"[\,\.\:\;]+(\w+)", r" \1", s)
        return s



class SpokenDialogueTokenizer:
    """
    Tokenizer that includes speaker identity embeddings. These embeddings are
    in the form of a speaker identity token for each input id.
    Ex: [id1, id2, id3 ..... idn]
    Let speaker tokens = [TOK-1, TOK-2]
    Then the speaker embedding: [TOK-1 TOK-1 TOK-2 .... ] indicates that the
    first two ids are by speaker 1 etc.

    Inspired by the AutoTokenizer class as well as the GPT-2 Tokenizer (links below).
    AutoTokenizer: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
    GPT-2 Tokenizer: https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/gpt2/tokenization_gpt2.py#L104

    NOTE: We do not 'build' a new tokenizer object because we do not want to train it
    - instead, we are simply wrapping a pre-trained GPT-2 Tokenizer that
    has special embeddings.
    """

    _TESTED_TOKENIZERS = ("gpt2")
    _LARGE_MODEL_MAX_LENGTH = 1e30

    def __init__(self,
            pretrained_model_name_or_path : str,
        ):
        if not pretrained_model_name_or_path in self._TESTED_TOKENIZERS:
            print(f"WARNING: Using untested tokenizer: {pretrained_model_name_or_path}")

        # Vars.
        self.normalizer = SpokenNormalizer()
        # -- Load tokenizer
        # NOTE: The AutoTokenizer itself contains its own normalizer, pre-tokenizer etc.
        self._tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        # Since we will construct the model - we set the maximum number of
        # input tokens the model can handle manually to a large integer
        self._tokenizer.model_max_length = self._LARGE_MODEL_MAX_LENGTH
        # Add speaker tokens
        self.sp1_token = BASE_SPEAKER_TOKEN.format("1")
        self.sp2_token = BASE_SPEAKER_TOKEN.format("2")
        # Adding new tokens to base tokenizer
        num_tokens_added = self._tokenizer.add_special_tokens({
            "eos_token" : TOKENIZER_EOS_TOKEN,
            "pad_token" : TOKENIZER_PAD_TOKEN,
            "additional_special_tokens" : [
                self.sp1_token, self.sp2_token,CONV_START_TOKEN, CONV_END_TOKEN]
            })
        msg = f"Special tokens added to tokenizer: {num_tokens_added}\n"
        msg = "Additional special tokens map:\n"
        for k,v in self._tokenizer.special_tokens_map.items():
            msg += f"\t{k}: {v}\n"
        print(msg)

    @property
    def unk_token(self):
        return self._tokenizer.unk_token

    @property
    def unk_token_id(self):
        return self._tokenizer.unk_token_id

    @property
    def eos_token(self):
        return self._tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def sp1_token_id(self):
        return self._tokenizer.convert_tokens_to_ids(self.sp1_token)

    @property
    def sp2_token_id(self):
        return self._tokenizer.convert_tokens_to_ids(self.sp2_token)

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self._tokenizer.encode(*args,**kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_tokens_to_string(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_string(*args, **kwargs).strip()

    def normalize(self, s : str):
        return self.normalizer.normalize_str(s)
        return s

    def __repr__(self):
        return self._tokenizer.__repr__()

    def __len__(self):
        return self._tokenizer.__len__()

    def __call__(self,
            text,
            add_prefix_space=True,
            add_eos_token=True,
            return_token_type_ids=True,
            **kwargs
        ):
        """
        Args:
            add_prefix_space (bool): If True, add a space before the first word.
                This is important because the tokenized representation in BPE for
                words with and without whitespaces is different.
            add_eos_token (bool): If True, add an end of sequence token to the
                end of the string.
            return_token_type_ids (bool): If True, return the ids of this specific
                tokenizer, which includes the 'speaker_ids'
        """
        if self._is_list_of_lists(text):
            ret = defaultdict(lambda : list())
            for t_list in text:
                output = self(
                    t_list,
                    add_prefix_space=add_prefix_space,
                    add_eos_token=add_eos_token,
                    return_token_type_ids=return_token_type_ids
                )
                for k,v in output.items():
                        ret[k].append(v)
            return dict(ret)
        elif self._is_list_of_strings(text):
            # List of strings gets combined into single larger string.
            dialog_string = " " if add_prefix_space else ""
            dialog_string += self.normalize(text[0])
            if len(text) > 1:
                dialog_string += self.eos_token
                for t_string in text[1:-1]:
                    dialog_string += " " + self.normalize(t_string) + self.eos_token
                dialog_string += " " + self.normalize(text[-1])
            if add_eos_token:
                dialog_string += self.eos_token
            text = dialog_string
        elif isinstance(text, str):
            text = self.normalize(text)
        else:
            raise  ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )
        # Obtain the base encodings
        encoding = self._tokenizer(text,**kwargs)
        # Add the speaker_ids embeddings
        if return_token_type_ids:
            encoding['speaker_ids'] = self._extract_speaker_states(
                input_ids=encoding['input_ids'])
        return encoding

    def _extract_speaker_states(self, input_ids):
        """
        For the given input ids, extracts the speaker ids for the corresponding
        input id.
        Assumptions:
            1. There are only two speaker.
            2. Each speaker turn ends with an end of sequence token.
            3. Speaker 1 appears before speaker 2.
        """
        is_input_batch = True
        if not isinstance(input_ids, torch.Tensor):
            is_input_batch = False
            # Convert to batch dimension.
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        # Initialize all ids to sp1
        speaker_ids = torch.ones_like(input_ids) * self.sp1_token_id
        batch, eos_idx = torch.where(input_ids == self.eos_token_id)
        for b in batch.unique():
            tmp_eos = eos_idx[batch == b]
            if len(tmp_eos) == 1:
                speaker_ids[b,eos_idx +1 : ] = self.sp2_token_id
            else:
                start = tmp_eos[0]
                for i, eos in enumerate(tmp_eos[1:]):
                    if i % 2 == 0:
                        speaker_ids[b,start+1:eos+1] = self.sp2_token_id
                    start = eos
                if i % 2 == 1:
                    speaker_ids[b,start+1:] = self.sp2_token_id
        if not is_input_batch:
            speaker_ids = speaker_ids.squeeze().tolist()
            if isinstance(speaker_ids,int):
                speaker_ids = [speaker_ids]
        return speaker_ids

    def _is_valid_text_input(t):
        if isinstance(t, str):
            # Strings are fine
            return True
        elif isinstance(t, (list, tuple)):
            # List are fine as long as they are...
            if len(t) == 0:
                # ... empty
                return True
            elif isinstance(t[0], str):
                # ... list of strings
                return True
            elif isinstance(t[0], (list, tuple)):
                # ... list with an empty list or with a list of strings
                return len(t[0]) == 0 or isinstance(t[0][0], str)
            else:
                return False
        else:
            return False

    def _is_list_of_lists(self, t):
        if isinstance(t, (list, tuple)):
            if isinstance(t[0], (list, tuple)):
                # ... list with an empty list or with a list of strings
                return len(t[0]) == 0 or isinstance(t[0][0], str)
        return False

    def _is_list_of_strings(self, t):
        if isinstance(t, (list, tuple)):
            if len(t) == 0:
                # ... empty
                return True
            if isinstance(t[0], str):
                # ... list of strings
                return True
        return False

