# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-20 10:58:43
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-20 15:41:02

import pytest
import sys
import os
from typing import Dict

from LLMDialogue.datasets.ICC import ICC, load_ICC


@pytest.mark.parametrize("config_name", ["labels", "no_labels"])
@pytest.mark.parametrize(
    "data_files",
    [
        {
            "train": "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/data/raw/ICC/5_train_37_test_set/train",
            "test": "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/data/raw/ICC/5_train_37_test_set/test",
        },
        {
            "train": "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/data/raw/ICC/28_train_14_test_set/train",
            "test": "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/data/raw/ICC/28_train_14_test_set/test",
        },
    ],
)
def test_create(config_name: str, data_files: Dict):
    dset = load_ICC(config_name=config_name, data_files=data_files)
