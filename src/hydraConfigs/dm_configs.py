# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-29 17:05:30
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 11:42:50

import hydra_zen
import sys
import os
from hydra_zen import hydrated_dataclass
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf
from hydra_zen import (
    builds,
    make_config,
    make_custom_builds_fn,
    instantiate,
    MISSING,
    store,
)
from LLMDialogue.tokenizers import SpokenDialogueTokenizer
from transformers import PreTrainedTokenizer
from .dataset_configs import (
    ICCLabeled28Train14TestConfig,
    ICCLabeled5Train37TestConfig,
    ICCUnlabeled28Train14TestConfig,
    ICCUnlabeled5Train37TestConfig,
)


#############
# BASE CONFIGS
#############


@hydrated_dataclass
class BaseDMConfigs:
    pass


#############
# ICC DM CONFIGS
#############


@hydrated_dataclass
class ICCHFLMDMConfigs(BaseDMConfigs):
    dset_variant: str = MISSING
    tokenizer: PreTrainedTokenizer = MISSING
    save_dir: str = MISSING
    data_block_size: int = 128
    batched: bool = True
    group_by_conversation: bool = False


@hydrated_dataclass
class ICCLabeledDM5Train37TestConfigs(
    ICCHFLMDMConfigs,
    ICCLabeled5Train37TestConfig,
):
    pass


@hydrated_dataclass
class ICCLabeledDM28Train14TestConfigs(
    ICCHFLMDMConfigs,
    ICCLabeled28Train14TestConfig,
):
    pass


@hydrated_dataclass
class ICCUnlabeledDM5Train37TestConfigs(
    ICCHFLMDMConfigs,
    ICCUnlabeled5Train37TestConfig,
):
    pass


@hydrated_dataclass
class ICCUnlabeledDM28Train14TestConfigs(
    ICCHFLMDMConfigs,
    ICCUnlabeled28Train14TestConfig,
):
    pass
