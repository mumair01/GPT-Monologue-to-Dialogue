# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 09:23:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-01 02:51:01

import sys
import os
from typing import List, Dict
import re
from venv import create

from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)

############### String utils #############

def add_whitespace_after_punctuation(s : str) -> str:
    """
    Add a whitespace after a punctuation in the given string.
    """
    return re.sub(r"[\,\.\:\;]+(\w+)", r" \1", s)

def remove_words_from_string(s : str, remove_words : List[str]) -> str:
    """
    Remove all occurrences of the given words from the string.
    """
    return " ".join([word for word in s.split() if word not in remove_words])

def replace_word_from_string(s : str, word : str, replacement : str):
    # print(s, word, replacement, re.sub(f"({word})",replacement, s))
    return re.sub(f"({word})",replacement, s)


############### Normalizers #############

def create_transformer_normalizer_sequence(
    lowercase : bool = True,
    unicode_normalizer : str = "nfd",
    strip_accents : bool = True,
    remove_punctuation : bool = True,
    remove_extra_whitespaces : bool = True,
):
    """
    Create a normalizer sequence that is compatible with huggingface transformers.
    """
    _UNICODE_NORMALIZERS = {
        "nfd" : NFD()
    }

    transformer_normalizers = []
    if lowercase:
        transformer_normalizers.append(Lowercase())
    if unicode_normalizer in _UNICODE_NORMALIZERS:
        transformer_normalizers.append(_UNICODE_NORMALIZERS[unicode_normalizer])
    else:
        raise NotImplementedError(
            f"Unicode normalizer {unicode_normalizer} for supported"
        )
    if strip_accents:
        transformer_normalizers.append(StripAccents())
    if remove_punctuation:
        transformer_normalizers.append(
            Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), "")
        )
    if remove_extra_whitespaces:
        transformer_normalizers.extend([
            Replace(Regex(r"\s\s+"), " "),  # double spaces
            Strip(),
        ])
    return Sequence(transformer_normalizers)


def create_normalizer_sequence(
    lowercase : bool = True,
    unicode_normalizer : str = "nfd",
    strip_accents : bool = True,
    remove_punctuation : bool = True,
    remove_extra_whitespaces : bool = True,
    add_whitespace_punc : bool = True,
    remove_words : List[str] = [],
    replace_words : Dict = {},
    custom_regex : str = None
):
    # Create the transformers normalizer sequence
    transformer_sequence = create_transformer_normalizer_sequence(
        lowercase=lowercase,
        unicode_normalizer=unicode_normalizer,
        strip_accents=strip_accents,
        remove_punctuation=remove_punctuation,
        remove_extra_whitespaces=remove_extra_whitespaces
    )

    def call(s : str):
        if not custom_regex is None:
            s = re.sub(custom_regex,"",s)

        if add_whitespace_punc:
            add_whitespace_after_punctuation(s)
        s = transformer_sequence.normalize_str(s)

        s = remove_words_from_string(s,remove_words)

        for w, replacement in replace_words.items():
            s = replace_word_from_string(s,w,replacement)
        return s

    return call
