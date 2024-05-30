# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-12-15 15:43:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-29 16:59:08


import os
import sys
from pathlib import Path
from LLMDialogue.utils import read_toml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any


def prepend_path_if_possible(data: Any, path_prefix: str) -> Any:
    """
    Go through all the contents of a dictionary (that may be nested)
    recursively and prepend the specified path if that makes it a full path.
    NOTE: This function assumes that anything that starts with 'cache' or
    'data' is a relative path and will make that into a full path. This is
    the only assumption we will make for the data dictionaries here.
    """
    if isinstance(data, str):
        # NOTE: Here are the assumptions...
        if data.startswith(("data/", "cache/")):
            # Combine with path prefix (e.g., script directory) if provided
            return os.path.join(path_prefix, data)
        else:
            return data
    elif isinstance(data, dict):
        return {
            key: prepend_path_if_possible(value, path_prefix)
            for key, value in data.items()
        }
    else:
        return data


# NOTE: Assuming that the configs dir is in root and that all the paths
# in the config files are relative to this root!
# Path to the root of the repo.
_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_MODULE_PATH = os.path.dirname(__file__)
_CONFIGS_DIR = os.path.join(_ROOT_PATH, "configs/base")

_DSET_CONFIG_FPATH = os.path.join(_CONFIGS_DIR, "DsetConfigs.toml")
_PIPELINES_CONFIG_FPATH = os.path.join(_CONFIGS_DIR, "pipelineConfigs.toml")

_DSET_CONFIGS = read_toml(_DSET_CONFIG_FPATH)
_PIPELINES_CONFIGS = read_toml(_PIPELINES_CONFIG_FPATH)


_DSET_CONFIGS = prepend_path_if_possible(_DSET_CONFIGS, _ROOT_PATH)
_PIPELINES_CONFIGS = prepend_path_if_possible(_PIPELINES_CONFIGS, _ROOT_PATH)


@dataclass
class BaseConfigs:
    root_path: str = _ROOT_PATH
    module_path: str = _MODULE_PATH

    @dataclass
    class datasets:
        cachePath: str = _DSET_CONFIGS["cache_dir"]

        configsFilePath: str = _DSET_CONFIG_FPATH

    @dataclass
    class pipeline:
        cachePath: str = _PIPELINES_CONFIGS["cache_dir"]
        configsFilePath: str = _PIPELINES_CONFIG_FPATH
