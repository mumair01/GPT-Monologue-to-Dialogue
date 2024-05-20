# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-20 10:58:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-20 15:45:30

import pytest
import sys
import os
from typing import Dict
from LLMDialogue.datasets.speakerIdentity import (
    SpeakerIdentityDataset,
    load_speakerIdentity,
)


@pytest.mark.parametrize("config_name", ["labels", "no_labels"])
def test_create(config_name: str):
    # NOTE: Data dir is being loaded from the config file implicitly.
    dset = load_speakerIdentity(config_name=config_name)
