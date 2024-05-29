# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-20 10:58:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-29 11:38:04

import pytest
import sys
import os
from typing import Dict
from LLMDialogue.datasets.speakerIdentity import (
    SpeakerIdentityDataset,
    load_speakerIdentity,
)


@pytest.mark.parametrize("config_name", ["labels", "no_labels"])
@pytest.mark.parametrize("force_regenerate", [True, False])
@pytest.mark.parametrize("add_eos_tok_bw_turns", [True, False])
def test_create(
    config_name: str,
    force_regenerate: bool,
    add_eos_tok_bw_turns: bool,
):
    # NOTE: Data dir is being loaded from the config file implicitly.
    dset = load_speakerIdentity(
        config_name=config_name,
        force_regenerate=force_regenerate,
        add_eos_tok_bw_turns=add_eos_tok_bw_turns,
    )
