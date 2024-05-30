# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 19:48:31
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 13:35:20


import sys
import os
from LLMDialogue.baseConfigs import BaseConfigs, prepend_path_if_possible
from LLMDialogue.utils.general import read_toml

from copy import deepcopy
from .ICC import ICCDataset, ICCDatasetConfig

import logging

logger = logging.getLogger(__name__)


_CONFIGS = read_toml(BaseConfigs.datasets.configsFilePath)

_CONFIGS = prepend_path_if_possible(_CONFIGS, BaseConfigs.root_path)

_BUILDER_CONFIGS = _CONFIGS["ICC"]["builderConfigs"]


_SUPPORTED_CONFIGS = ("no_labels", "labels")

assert all([k in _SUPPORTED_CONFIGS for k in _BUILDER_CONFIGS.keys()])


# TODO: Later, add a way to pass kwargs that can overwrite the ones that are in the builder configs.
def load_ICC(config_name: str, **kwargs) -> ICCDataset:
    assert config_name in list(
        _BUILDER_CONFIGS.keys()
    ), f"{config_name} not in {list(_BUILDER_CONFIGS.keys())}"
    conf_dict = deepcopy(_BUILDER_CONFIGS[config_name])
    conf_dict.update(kwargs)
    config = ICCDatasetConfig(**conf_dict)
    dset = ICCDataset(config)
    return dset
