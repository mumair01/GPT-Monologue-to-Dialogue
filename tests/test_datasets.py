# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-01 02:32:17
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-03 01:25:02

import sys
import os
import pytest

from data_lib.icc import ICCDataset
from data_lib.speaker_identity_stims import SpeakerIdentityStimuliDataset


################################## GLOBALS ###################################

ROOT_PATH = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue"
OUTPUT_DIR = "./tests/output/test_datasets_output"

FIVE_TRAIN_THIRTY_SEVEN_TEST_DIR_PATH = "data/raw/ICC/finetune_experiments_julia/5_train_37_test_set/test"
FIVE_TRAIN_THIRTY_SEVEN_TRAIN_DIR_PATH = "data/raw/ICC/finetune_experiments_julia/5_train_37_test_set/train"

TWENTY_EIGHT_TRAIN_FOURTEEN_TEST_PATH = "data/raw/ICC/finetune_experiments_julia/28_train_14_test_set/test"
TWENTY_EIGHT_TRAIN_FOURTEEN_TRAIN_PATH = "data/raw/ICC/finetune_experiments_julia/28_train_14_test_set/train"

SPEAKER_IDENTITY_SIMTS_PATH = "data/raw/speaker_identity_stims"

############################### TESTING METHODS ##############################


@pytest.mark.parametrize("variant, output_dir, data_dir", [
   ("special_labels", OUTPUT_DIR, FIVE_TRAIN_THIRTY_SEVEN_TEST_DIR_PATH),
   ("no_labels", OUTPUT_DIR, FIVE_TRAIN_THIRTY_SEVEN_TEST_DIR_PATH)
])
def test_icc_dataset_call(variant, output_dir, data_dir):
    dataset = ICCDataset(data_dir)
    dataset(variant, output_dir, os.path.basename(data_dir))


@pytest.mark.parametrize("variant, output_dir, data_dir", [
   ("special_labels", OUTPUT_DIR, SPEAKER_IDENTITY_SIMTS_PATH),
#    ("no_labels", OUTPUT_DIR, SPEAKER_IDENTITY_SIMTS_PATH)
])
def test_speaker_identity_dataset_call(variant, output_dir, data_dir):
    dataset = SpeakerIdentityStimuliDataset(data_dir)
    dataset(variant,output_dir)
