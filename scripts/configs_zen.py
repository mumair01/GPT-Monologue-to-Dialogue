# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-24 12:53:14
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-24 12:54:15

"""
Here, we will define the configurations that we need for our experiments / scripts
as structured configs using hydra-zen. 
"""

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
