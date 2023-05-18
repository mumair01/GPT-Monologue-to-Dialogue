# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-15 17:30:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-18 09:46:52


import pytest
import sys
import os

import transformers
import pytorch_lightning as pl
import numpy as np

from gpt_dialogue.turngpt.tokenizer import (
    SpokenNormalizer,
    SpokenDialogueTokenizer,
)
from gpt_dialogue.turngpt import TurnGPT
from gpt_dialogue.turngpt.dm import TurnGPTFinetuneDM

from tests.utils import load_configs

ROOT_PATH = os.getenv("PROJECT_PATH")
TRAIN_CSV_PATH = "data/processed/ICC/5_train_37_test/train_no_labels.csv"
VAL_CSV_PATH = "data/processed/ICC/5_train_37_test/test_no_labels.csv"
OUTPUT_DIR = "./tests/output/test_turngpt_finetune_dm"


def test_prepare_data():
    tokenizer = SpokenDialogueTokenizer(pretrained_model_name_or_path="gpt2")
    dm = TurnGPTFinetuneDM(
        tokenizer=tokenizer,
        train_csv_path=TRAIN_CSV_PATH,
        val_csv_path=VAL_CSV_PATH,
        conversation_id_key="convID",
        utterance_key="Utterance",
        save_dir=OUTPUT_DIR,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    train_dataloader = dm.train_dataloader()

    for item in train_dataloader:
        print(item["input_ids"].tolist())
        for item in item["input_ids"]:
            decoded = tokenizer.decode(item.tolist())
            print(f"Decoded: {decoded}")
