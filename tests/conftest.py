# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 18:18:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-19 19:16:24


import sys
import os
import pytest

from typing import List, Dict, Any, Callable

from tests.test_utils import load_toml

# Load the configs data dictionaries.

ROOT_PATH = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue"
MODELS_CONF_PATH = "configs/tests/models.toml"
PATHS_CONF_PATH = "configs/tests/paths.toml"


@pytest.fixture(scope="session", autouse=True)
def root_path():
    return ROOT_PATH


@pytest.fixture(scope="session", autouse=True)
def model_configs(root_path: str) -> Dict:
    return load_toml(os.path.join(root_path, MODELS_CONF_PATH))


@pytest.fixture(scope="session", autouse=True)
def paths_configs(root_path: str) -> Dict:
    return load_toml(os.path.join(root_path, PATHS_CONF_PATH))
