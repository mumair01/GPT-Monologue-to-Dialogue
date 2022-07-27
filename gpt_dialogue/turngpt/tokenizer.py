# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 10:26:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-27 14:41:59

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

from transformers import AutoTokenizer
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


TOKENIZER_PAD_TOKEN = "<PAD>"
TOKENIZER_EOS_TOKEN = "<|endoftext|>"
BASE_SPEAKER_TOKEN = "<SP{}"
ADDITIONAL_TOKENIZER_TOKENS = {
    "eos_token" : TOKENIZER_EOS_TOKEN,
    "pad_token" : TOKENIZER_PAD_TOKEN,
    "additional_special_tokens" : [
        BASE_SPEAKER_TOKEN.format("1"),
        BASE_SPEAKER_TOKEN.format("2"),
        "<START>",
        "<END>"
    ]
}

class SpokenNormalizer:
    """Normalizer to preprocess text (e.g, clear punctuation etc.)"""

    def __init__(self):
        self.normalizer = SpokenNormalizer.build_normalizer()

    def normalize_string(self, s):
        s = self.add_whitespace_after_punctuation(s)
        return self.normalizer.normalize_str(s)

    def add_whitespace_after_punctuation(self, s):
        s = re.sub(r"[\,\.\:\;]+(\w+)", r" \1", s)
        return s

    @staticmethod
    def build_normalizer():
        normalizer = Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),  # punctuation
                Replace(Regex(r"\s\s+"), " "),  # double spaces
                Strip(),
            ]
        )
        return normalizer

class SpokenDialogTokenizer(SpokenNormalizer):
    """
    Normalizer for Dialogue i.e., assumes that there are two speakers in the
    conversation only.
    Basically, the speaker token `embedding` is an array of speaker tokens
    corresponding to the input ids where a speaker identity token is assigned
    per speaker id.
    """
    _TOKENIZERS = [
        "gpt2"
    ]

    def __init__(self, pretrained_tokenizer_name_or_path : str, normalization : bool = True):
        super().__init__()
        self.name_or_path = pretrained_tokenizer_name_or_path
        self.normalization = normalization
        if not pretrained_tokenizer_name_or_path in self._TOKENIZERS:
            print(f"WARNING: Using untested tokenizer: {pretrained_tokenizer_name_or_path}")

        # Loading the pretrained tokenzier.
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_name_or_path, max_model_input_sizes=None
        )

        # Set to large number to avoid warnings
        # Manually keep track of your models maximum input length
        self._tokenizer.model_max_length = 1e30

        # Logging the number of additional tokens
        num_added_toks = self._tokenizer.add_special_tokens(ADDITIONAL_TOKENIZER_TOKENS)

        s = "Tokenizer initialization:\n"
        s += f"\tWe added {num_added_toks} tokens -> Special token map\n"
        for k, v in self._tokenizer.special_tokens_map.items():
            s += f"\t{k}: {v}\n"
        logger.info(s)

        # Generate the speaker tokens
        self.sp1_token = BASE_SPEAKER_TOKEN.format("1")
        self.sp2_token = BASE_SPEAKER_TOKEN.format("2")
        self.sp1_token_id = self._tokenizer.convert_tokens_to_ids(self.sp1_token)
        self.sp2_token_id = self._tokenizer.convert_tokens_to_ids(self.sp2_token)

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

    def __repr__(self) -> str:
        return self._tokenizer.__repr__()

    def __len__(self):
        return len(self._tokenizer)


    def __call__(self, text, return_token_type_ids=True, include_pre_spaces=False,
            include_end_ts=False, **kwargs) -> BatchEncoding:
        """
        NOTE: In this re-implementation, text will be a simple string or list
        # of strings.
        Args:
            text (list or list of lists or string)
            return_token_type_ids (bool):
                If True, return speaker identity tokens as separate entities.
            include_pre_spaces (bool): If True, add spaces b/w the tokens.
            include_end_ts (bool): If True, include eos token at the of sentence.
        """
        # Case 1: List of lists
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], list):
            ret = defaultdict(lambda : list())
            for text_list in text:
                output = self(
                    text_list,
                    return_token_type_ids=return_token_type_ids,
                    include_pre_spaces=include_pre_spaces,
                    include_end_ts=include_end_ts)
                for k,v in output.items():
                    ret[k].append(v)
            return ret
        # Case 2: List of strings
        elif isinstance(text, list):
            assert len(text) > 0
            dialog_string = " " if include_pre_spaces else ""
            dialog_string+=self.normalize(text[0])
            if len(text) > 1:
                dialog_string += self.eos_token
                for text_string in text[1:-1]:
                    dialog_string += " " + self.normalize(text_string) + self.eos_token
                dialog_string += " " + self.normalize(text[-1])
            if include_end_ts:
                dialog_string += self.eos_token
            text = dialog_string
        else:
            text = self.normalize(text)

        encoding = self._tokenizer(text, **kwargs)
        if return_token_type_ids:
            encoding["speaker_ids"] = self._extract_speaker_states(
                encoding["input_ids"])
        return encoding

    def _extract_speaker_states(self, input_ids):
        # extract speaker states
        back_to_list = False
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).unsqueeze(0)  # with batch dim
            back_to_list = True
        # initialize with speaker 1
        speaker_ids = torch.ones_like(input_ids) * self.sp1_token_id
        batch, eos_idx = torch.where(input_ids == self.eos_token_id)
        for b in batch.unique():
            tmp_eos = eos_idx[batch == b]
            if len(tmp_eos) == 1:
                speaker_ids[b, eos_idx + 1 :] = self.sp2_token_id
            else:
                start = tmp_eos[0]
                for i, eos in enumerate(tmp_eos[1:]):
                    if i % 2 == 0:
                        sp = self.sp2_token_id
                        speaker_ids[b, start + 1 : eos + 1] = sp
                    start = eos
                if i % 2 == 1:  # add sp2 tokens after last eos if i is odd
                    speaker_ids[b, start + 1 :] = self.sp2_token_id

        if back_to_list:
            speaker_ids = speaker_ids.squeeze().tolist()
            if isinstance(speaker_ids, int):
                speaker_ids = [speaker_ids]

        return speaker_ids

    def idx_to_tokens(self, ids):
        def list_ids_to_string(ids):
            return [
                self.convert_tokens_to_string(t)
                for t in self.convert_ids_to_tokens(ids)
            ]

        # tokenize keep tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, list):
            if isinstance(ids[0], list):
                ret = [list_ids_to_string(ids_list) for ids_list in ids]
            else:
                ret = list_ids_to_string(ids)
        else:
            ret = self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))
        return ret

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_tokens_to_string(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_string(*args, **kwargs).strip()

    def normalize(self, string : str):
        if self.normalization:
            return self.normalize_string(string)
        return string

if __name__ == "__main__":

    TOKENIZER_CHECKPOINT = "gpt2"
    TRAIN_PATH = ""
    # Load tokenizer
    tokenizer = SpokenDialogTokenizer(TOKENIZER_CHECKPOINT)

    # Load my dataset
    dataset = load_dataset("csv", data_files={
            'train' : '/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/data/datasets/processed/ICC/julia_dissertation/train.csv'
        })
    # Remove the speaker labels that were already there.
    def clean_speaker_labels(data):
        if len(data['Utterance'].split()) > 1:
            data['Utterance'] = " ".join(data['Utterance'].split()[1:-1])
        return data
    dataset['train'] = dataset['train'].map(
        clean_speaker_labels,
        remove_columns=["Unnamed: 0","convName","convID"])

    # The tokenizer expects a list of strings to allow it to treat each separate
    # list as an utterance by the next speaker.
    # NOTE: The assumption here is that each item in the list is an utterance
    # by another speaker and speaker 1 starts.
    train_dataset = [data['Utterance'] for data in list(dataset['train'])]
    # Tokenize this dataset
    tokenized_train_dataset = tokenizer(train_dataset)
    # # Tokenize the dataset.
    # def tokenize_fn(tokenizer):
    #     return lambda data: tokenizer(data["Utterance"], truncation=True)
    # tokenized_datasets = train_dataset.map(
    #         tokenize_fn(tokenizer),
    #         batched=False)
    print(tokenized_train_dataset)
    # for i in range(10):
    #     print(tokenized_train_dataset[i])
    #     print(tokenizer.decode(tokenized_train_dataset[i]['input_ids']))

    # # List of lists
    # out = tokenizer(
    #     [["hello", "bye"], ["hello", "bye", "you"]],
    #     return_token_type_ids=False,
    #     include_end_ts=False,
    #     include_pre_spaces=False)
    # print(out)
    # # List of strings
    # # out = tokenizer(["hi my name is Umair.", "That's not a great name","Well, I'm an intellectual bro."])
    # # print(out)
    # print(out['input_ids'])
    # print(tokenizer.decode(out['input_ids'][1]))