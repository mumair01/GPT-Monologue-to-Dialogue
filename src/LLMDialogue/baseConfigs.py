# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-12-15 15:43:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-22 08:05:35


import os
import sys
from pathlib import Path
from LLMDialogue.utils import read_toml
import json
from dataclasses import dataclass, asdict


# Path to the root of the repo.
_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_MODULE_PATH = os.path.dirname(__file__)
_CONFIGS_DIR = os.path.join(_ROOT_PATH, "configs")
_DATASETS_CONFIG_DIR = os.path.join(_CONFIGS_DIR, "datasets")
_PIPELINES_CONFIG_DIR = os.path.join(_CONFIGS_DIR, "pipelines")

_DSET_CONFIG_FPATH = os.path.join(_DATASETS_CONFIG_DIR, "DsetConfigs.toml")
_PIPELINES_CONFIG_FPATH = os.path.join(_PIPELINES_CONFIG_DIR, "pipelineConfigs.toml")

_DSET_CONFIGS = read_toml(_DSET_CONFIG_FPATH)
_PIPELINES_CONFIGS = read_toml(_PIPELINES_CONFIG_FPATH)


@dataclass
class BaseConfigs:
    root_path: str = _ROOT_PATH
    module_path: str = _MODULE_PATH

    @dataclass
    class datasets:
        cachePath: str = _DSET_CONFIGS["cache_dir"]
        configsDirPath: str = _DATASETS_CONFIG_DIR
        configsFilePath: str = _DSET_CONFIG_FPATH

    @dataclass
    class pipeline:
        cachePath: str = _PIPELINES_CONFIGS["cache_dir"]
        configsDirPath: str = _PIPELINES_CONFIG_DIR
        configsFilePath: str = _PIPELINES_CONFIG_FPATH
