# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-01 02:31:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-01 02:51:04


import pytest

from data_lib.core.normalize import (
    add_whitespace_after_punctuation,
    remove_words_from_string,
    replace_word_from_string,
    create_transformer_normalizer_sequence,
    create_normalizer_sequence
)


@pytest.mark.parametrize("s, remove_words, expected", [
    ("<SP1> needs to be replaced!!!", ["<SP1>", "needs"],"to be replaced!!!")
])
def test_remove_words_from_string(s, remove_words, expected):
    s2 = remove_words_from_string(
        s = s,
        remove_words=remove_words
    )
    assert s2 == expected

@pytest.mark.parametrize("s, word, replacement, expected", [
    ("<SP1> this is a turn", "<SP1>", "sp1" , "sp1 this is a turn"),
])
def test_replace_word_from_string(s, word, replacement, expected):
    s2 = replace_word_from_string(
        s=s,
        word=word,
        replacement=replacement
    )
    assert s2 == expected

@pytest.mark.parametrize("s, expected", [
    ("This is a very !!, cool test string","this is a very cool test string"),
])
def test_create_normalizer_sequence(s, expected):
    seq = create_normalizer_sequence()
    assert seq(s) == expected




# s = remove_words_from_string(
#         s = "<SP1> needs to be replaced!!!",
#         remove_words=["<SP1>", "needs"]
#     )
#     print(s)
#     print(replace_word_from_string("<SP1> this is a turn", "<SP1>", "sp1" ))



#     seq = create_normalizer_sequence()
#     print(seq(
#         "This is a very !!, cool test string"
#     ))






