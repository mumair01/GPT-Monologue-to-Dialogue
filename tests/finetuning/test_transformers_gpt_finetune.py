# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-08 11:11:38
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-08 11:36:45

import pytest
import os
import yaml

from src.finetuning.transformers_gpt_finetune import Configs, parse_configs,config_env,finetune

# PATHS
PROJECT_ROOT_PATH = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue"
CONFIGS_PATH = os.path.join(PROJECT_ROOT_PATH,"configs","finetuning")

def test_parse_configs():
    CONFIG_PATH = os.path.join(CONFIGS_PATH,"3.0-GPT-Finetune_TextDataset-Local.yaml")
    with open(CONFIG_PATH,"r") as f:
        configs_data = yaml.safe_load(f)
        configs= parse_configs(configs_data)
    assert type(configs) == Configs

def test_config_env():
    CONFIG_PATH = os.path.join(CONFIGS_PATH,"3.0-GPT-Finetune_TextDataset-Local.yaml")
    configs = config_env(CONFIG_PATH)
    assert type(configs) == Configs
