# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-27 10:26:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-16 10:01:06

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

import numpy as np
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


class SpokenTokenizer:
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

    This tokenizer is a base class for tokenizer that can specify the number of speakers.
    """

    _TESTED_TOKENIZERS = ("gpt2")
    _LARGE_MODEL_MAX_LENGTH = 1e30

    def __init__(self,
            pretrained_model_name_or_path : str,
            num_speakers : int,
            base_speaker_token : str,
            tokenizer_eos_token : str,
            tokenizer_pad_token : str,
            tokenizer_additional_special_tokens : List[str] = [],
        ):
        if not pretrained_model_name_or_path in self._TESTED_TOKENIZERS:
            print(f"WARNING: Using untested tokenizer: {pretrained_model_name_or_path}")
        assert num_speakers > 0, f"ERROR: Invalid number of speakers: {num_speakers}"

        # Vars.
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_eos_token = tokenizer_eos_token
        self.tokenizer_pad_token = tokenizer_pad_token
        self.tokenizer_additional_special_tokens = tokenizer_additional_special_tokens
        self.base_speaker_token = base_speaker_token
        self.num_speakers = num_speakers

        self._normalizer = SpokenNormalizer()
        # -- Load tokenizer
        # NOTE: The AutoTokenizer itself contains its own normalizer, pre-tokenizer etc.
        self._tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, max_model_input_sizes=None)
        # Since we will construct the model - we set the maximum number of
        # input tokens the model can handle manually to a large integer
        self._tokenizer.model_max_length = self._LARGE_MODEL_MAX_LENGTH

        # Create speaker tokens based on the number of speakers.
        self.speaker_tokens_map = {
            i + 1 : self.base_speaker_token.format(i + 1) for i in range(num_speakers)
        }
        self.tokens_speaker_map = dict(zip(
            self.speaker_tokens_map.values(), self.speaker_tokens_map.keys()
        ))


        num_tokens_added = self._tokenizer.add_special_tokens({
            "eos_token" : tokenizer_eos_token,
            "pad_token" : tokenizer_pad_token,
            "additional_special_tokens" : self.tokenizer_additional_special_tokens + \
                list(self.speaker_tokens_map.values())
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
    def pad_token(self):
        return self._tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

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

    def convert_id_to_string(self, id):
        """Convert an ID to a string"""
        if isinstance(id, torch.Tensor):
            id = id.item()
        return self.convert_tokens_to_string(
            self.convert_ids_to_tokens(id).strip())

    def normalize(self, s : str):
        return self._normalizer.normalize_str(s)

    def __repr__(self):
        return self._tokenizer.__repr__()

    def __len__(self):
        return self._tokenizer.__len__()

    # TODO: start_speaker starts from 1 - should add assert somewhere.
    def __call__(self,
            text,
            add_prefix_space=True,
            add_eos_token=True,
            return_token_type_ids=True,
            split_speaker_by_inline_eos=False,
            start_speaker=1,
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
            split_speaker_by_inline_eos (bool):
                If True, for every string, changes the speaker token assigned
                when an EOS token is encountered. If False, then the speaker
                is not changed after the EOS token. This is useful if we
                want to indicate multiple turns by the same speaker.
            start_speaker (int):
                The number of the speaker that should be assigned first. Must
                be greater than 0.
        """

        assert 0 < start_speaker <= self.num_speakers, \
            f"ERROR: Start speaker should be in range [0, {self.num_speakers}]"

        if self._is_list_of_lists(text):
            ret = defaultdict(lambda : list())
            for t_list in text:
                output = self(
                    t_list,
                    add_prefix_space=add_prefix_space,
                    add_eos_token=add_eos_token,
                    return_token_type_ids=return_token_type_ids,
                    split_speaker_by_inline_eos=split_speaker_by_inline_eos,
                    start_speaker=start_speaker,
                    **kwargs
                )
                for k,v in output.items():
                    # ret[k].append(torch.tensor(v))
                    ret[k].append(list(v))
            # NOTE: Moving padding from model.tokenize_strings to here
            ret = dict(ret)
            if isinstance(ret['input_ids'], list):
                temp_inp = []
                temp_sp = []
                for inp, sp in zip(ret['input_ids'], ret['speaker_ids']):
                    # temp_inp.append(torch.tensor(inp))
                    # temp_sp.append(torch.tensor(sp))
                    temp_inp.append(list(inp))
                    temp_sp.append(list(sp))
                temp = self._tokenizer.pad({"input_ids" : temp_inp})
                ret['input_ids'] = temp['input_ids']
                ret['attention_mask'] = temp['attention_mask']
                # NOTE: IMPORTANT: The speaker ID PAD value is the same as the
                # input_ids PAD value.
                ret['speaker_ids'] = self._tokenizer.pad({
                    "input_ids" : temp_sp})['input_ids']
            return ret
        elif self._is_list_of_strings(text):
            # List of strings gets combined into single larger string with
            # eos token between the strings.
            ret = defaultdict(lambda : list())
            for t in text:
                # The start speaker is the one after the previous speaker
                if len(ret["speaker_ids"]) > 0:
                    if type(ret["speaker_ids"][-1]) == torch.Tensor:
                        prev_speaker = ret["speaker_ids"][-1][-1].item()
                    else:
                        prev_speaker = ret["speaker_ids"][-1]
                    start_speaker = (self.tokens_speaker_map[self.convert_ids_to_tokens(
                        prev_speaker
                    )] % self.num_speakers) + 1
                output = self.__call__(
                    t,
                    add_prefix_space=add_prefix_space,
                    add_eos_token=add_eos_token,
                    return_token_type_ids=return_token_type_ids,
                    split_speaker_by_inline_eos=split_speaker_by_inline_eos,
                    start_speaker=start_speaker,
                    **kwargs
                )
                for k,v in output.items():
                    if type(v) == torch.Tensor:
                        ret[k] = torch.cat((torch.as_tensor(ret[k],dtype=torch.int64),v),dim=-1)
                    elif type(v) == list:
                        ret[k].extend(v)
                    else:
                        raise NotImplementedError(
                            f"ERROR: Cannot merge types: {type(k[v])} and {type(v)}"
                        )
            ret = BatchEncoding(ret)
            return ret

        elif isinstance(text, str):
            # Extract the speaker states here per string
            prefix = " " if add_prefix_space else ""
            text = prefix + self.normalize(text)
            if add_eos_token:
                text += self.eos_token
            encoding = self._tokenizer(text, **kwargs)
            # Add the speaker_ids embeddings
            if return_token_type_ids:
                encoding['speaker_ids'] = self._extract_speaker_states(
                    input_ids=encoding['input_ids'],
                    split_speaker_by_inline_eos=split_speaker_by_inline_eos,
                    start_speaker=start_speaker
                )
            return encoding
        else:
            raise  ValueError(
                "text input must of type `str` (single example), `List[str]` "
                "(batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )


    def _extract_speaker_states(
            self,
            input_ids,
            split_speaker_by_inline_eos : bool,
            start_speaker : int
        ):
        """
        For the given input ids, extracts the speaker ids for the corresponding
        input id.
        """
        assert 0 < start_speaker <= self.num_speakers, \
            f"ERROR: Start speaker should be in range [0, {self.num_speakers}]"

        is_input_batch = True
        if not isinstance(input_ids, torch.Tensor):
            is_input_batch = False
            # Convert to batch dimension.
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Initialize all ids to the specified speaker in mapping
        speaker_ids = torch.ones_like(input_ids) * self._tokenizer.convert_tokens_to_ids(
            self.speaker_tokens_map[start_speaker])
        if split_speaker_by_inline_eos:
            batch, eos_idx = torch.where(input_ids == self.eos_token_id)
        else:
            # This assumes that all of the input ids speaker states will NOT
            # be split by the eos token.
            batch = torch.tensor([0])
            eos_idx = torch.tensor([input_ids.size()[1]])

        # TODO: Eventually test this more but I think this generalizes to any number of speakers.
        for b in batch.unique():
            tmp_eos = eos_idx[batch == b]
            start = 0
            for i, eos in enumerate(tmp_eos):
                speaker_map_key = ((i + start_speaker -1)  % self.num_speakers) + 1
                speaker_ids[b,start+1:eos+1] = \
                        self._tokenizer.convert_tokens_to_ids(self.speaker_tokens_map[speaker_map_key])
                start = eos
            # Add speaker id to last sentence
            speaker_map_key = ((i + start_speaker)  % self.num_speakers) + 1
            speaker_ids[b,tmp_eos[-1] + 1 :] = \
                self._tokenizer.convert_tokens_to_ids(self.speaker_tokens_map[speaker_map_key])

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


class SpokenDialogueTokenizer(SpokenTokenizer):
    """SpokenTokenizer for dialogue i.e., with two speakers"""

    _NUM_SPEAKERS = 2

    def __init__(
            self,
            pretrained_model_name_or_path : str = "gpt2",
            base_speaker_token : str = "<SP{}>",
            tokenizer_eos_token : str = "<ts>",
            tokenizer_pad_token : str = "<|endoftext|>",
            tokenizer_additional_special_tokens : List[str] = []
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_speakers=self._NUM_SPEAKERS,
            base_speaker_token=base_speaker_token,
            tokenizer_eos_token=tokenizer_eos_token,
            tokenizer_pad_token=tokenizer_pad_token,
            tokenizer_additional_special_tokens=tokenizer_additional_special_tokens
        )
        print(f"NOTE: This tokenizer assumes that there are {self._NUM_SPEAKERS} speakers")

    @property
    def sp1_token(self):
        return self.speaker_tokens_map[1]

    @property
    def sp2_token(self):
        return self.speaker_tokens_map[2]

    @property
    def sp1_token_id(self):
        return self._tokenizer.convert_tokens_to_ids(self.sp1_token)

    @property
    def sp2_token_id(self):
        return self._tokenizer.convert_tokens_to_ids(self.sp2_token)

    def decode_speaker(self, toks, speaker, *args, **kwargs) -> List:
        """
        Decode the turns by the specified speaker.
        Note that this only works if toks contains lists, not tensors.
        """

        if len(np.asarray(toks["speaker_ids"]).shape) == 2:
            decoded = []
            for input_ids, speaker_ids, attention_mask in zip(
                toks["input_ids"], toks["speaker_ids"], toks["attention_mask"]
            ):
                decoded.append(
                    self._decode_speaker_toks({
                        "input_ids" : input_ids,
                        "speaker_ids" : speaker_ids,
                        "attention_mask" : attention_mask
                    }, speaker)
                )
            return decoded
        else:
            return [self._decode_speaker_toks(toks, speaker)]

    def _decode_speaker_toks(self, toks, speaker):
        sp_token_id = self.sp1_token_id if speaker == 1 else self.sp2_token_id
        sp_idx = np.where(np.asarray(toks['speaker_ids']) == sp_token_id)
        sp_input_ids = np.take(toks['input_ids'], sp_idx)
        return self.decode(*sp_input_ids)


if __name__ == "__main__":

    # NOTE: In this case, ts is the string that is used as the EOS token
    # in the tokenizer.
    # conv = ["i tripped in front of my boss at work today<ts> don't laugh"]
    conv =  [
        [
            "i tripped in front of my boss at work today<ts> don't laugh",
            "that is really funny",
            "i don't think that was funny<ts> did you",
            "hello one two three"
        ],
        [
            "the cat is on the<ts> the mat",
            "no it is not"
        ]
    ]
    tokenizer = SpokenDialogueTokenizer(
        pretrained_model_name_or_path="gpt2"
    )
    output = tokenizer(
        text=conv,
        split_speaker_by_inline_eos=False,
        start_speaker = 1
    )
    sp1_decoded = tokenizer.decode_speaker(output, 1)
    sp2_decoded = tokenizer.decode_speaker(output, 2)

    print(f"Decoded for speaker 1:")
    for item in sp1_decoded:
        print(item)

    print(f"Decoded for speaker 2:")
    for item in sp2_decoded:
        print(item)

