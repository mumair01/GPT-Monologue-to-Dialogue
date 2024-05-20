# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-12-15 15:43:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-20 11:05:44


import os
import sys
from pathlib import Path
from LLMDialogue.utils import read_toml
import json
from dataclasses import dataclass, asdict


# Path to the root of the repo.
_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
_MODULE_PATH = os.path.dirname(__file__)
_CONFIGS_DIR = os.path.join(_ROOT_PATH, "configs/datasets")
_CONFIG_FPATH = os.path.join(_CONFIGS_DIR, "DsetConfigs.toml")
_CONFIGS = read_toml(_CONFIG_FPATH)
_CACHE_DIR = _CONFIGS["cache_dir"]


@dataclass
class PkgPaths:
    @dataclass
    class Core:
        rootPath: str = _ROOT_PATH
        modulePath: str = _MODULE_PATH
        cachePath: str = _CACHE_DIR
        configsDirPath: str = _CONFIGS_DIR
