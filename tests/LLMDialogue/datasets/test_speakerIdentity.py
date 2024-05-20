# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-20 10:58:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-20 11:07:10


import sys
import os
from LLMDialogue.datasets.speakerIdentity import (
    SpeakerIdentityDataset,
    load_speakerIdentity,
)


def test_create():
    dset = load_speakerIdentity(config_name="default")
