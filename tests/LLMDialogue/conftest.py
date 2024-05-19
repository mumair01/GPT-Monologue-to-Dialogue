# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 18:28:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-19 19:18:59

import sys
import os
import pytest

from typing import List, Dict, Any, Callable
from LLMDialogue.models import GPT2, TurnGPT
from LLMDialogue.pipelines import ConditionalProbabilityPipeline
from LLMDialogue.tokenizers import SpokenDialogueTokenizer

from tests.test_utils import load_inference_text_from_file, prepend_to_str_dict

#############
# OBJECT CONFIG FIXTURES
#############


@pytest.fixture(scope="session", autouse=True)
def gpt2_load_configs(model_configs):
    return model_configs["gpt2"]["load"]


@pytest.fixture(scope="session", autouse=True)
def turngpt_load_configs(model_configs):
    return model_configs["turngpt"]["load"]


@pytest.fixture(scope="session", autouse=True)
def SD_tokenizer_configs(model_configs):
    return model_configs["spoken_dialogue_tokenizer"]


@pytest.fixture(scope="session", autouse=True)
def conditionalProb_pipe_configs(model_configs):
    return model_configs["conditional_probability_pipe"]


#############
# OBJECT  FIXTURES
#############


@pytest.fixture(scope="function")
def gpt2_model(gpt2_load_configs):
    model = GPT2()
    model.load(**gpt2_load_configs)
    return model


@pytest.fixture(scope="function")
def turngpt_model(turngpt_load_configs):
    model = GPT2()
    model.load(**turngpt_load_configs)
    return model


@pytest.fixture(scope="function")
def SD_tokenizer(SD_tokenizer_configs):
    tokenizer = SpokenDialogueTokenizer(**SD_tokenizer_configs)
    return tokenizer


#############
# PATH FIXTURES
#############


@pytest.fixture(scope="session", autouse=True)
def results_path(root_path, paths_configs):
    return os.path.join(root_path, paths_configs["paths"]["results"])


@pytest.fixture(scope="session", autouse=True)
def inference_text_path(root_path, paths_configs):
    return os.path.join(root_path, paths_configs["data_files"]["inferenceText"])


@pytest.fixture(scope="session", autouse=True)
def ICC_5_train_37_test_fpaths(root_path, paths_configs) -> Dict:
    return prepend_to_str_dict(
        root_path, paths_configs["data_files"]["icc"]["5_train_37_test"]
    )


# @pytest.fixture(scope="session", autouse=True)
# def ICC_28_train_14_test_fpaths(root_path, paths_configs) -> Dict:
#     return os.path.join(
#         root_path, paths_configs["data_files"]["icc"]["28_train_14_test"]
#     )


#############
# DATA FIXTURES
#############


@pytest.fixture(scope="session", autouse=True)
def inference_text(inference_text_path):
    return load_inference_text_from_file(inference_text_path)
