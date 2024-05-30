# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-24 12:53:14
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-29 12:53:02

"""
Here, we will define the configurations that we need for our experiments / scripts
as structured configs using hydra-zen. 
"""

import sys
import os
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
from LLMDialogue.baseConfigs import BaseConfigs

##################
# GLOBALS
##################

DATA_FILE_PATHS = {
    "28_train_14_test": {
        "train": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/28_train_14_test_set/train"
        ),
        "test": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/28_train_14_test_set/test"
        ),
    },
    "5_train_37_test": {
        "train": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/5_train_37_test_set/train"
        ),
        "test": os.path.join(
            BaseConfigs.root_path, "data/raw/ICC/5_train_37_test_set/test"
        ),
    },
}


###############
# Custom Builds
###############
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


##################
# TRACKERS
##################

WANDBTrackerConf = pbuilds()


##################
# DATASETS
##################
