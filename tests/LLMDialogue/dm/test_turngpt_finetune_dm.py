# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-15 17:30:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-19 19:17:07


import pytest
import sys
import os

import transformers
import pytorch_lightning as pl
import numpy as np

from LLMDialogue.tokenizers import SpokenDialogueTokenizer
from LLMDialogue.dm import TurnGPTFinetuneDM


def test_prepare_data(
    results_path,
    ICC_5_train_37_test_fpaths,
):
    train_csv_path = ICC_5_train_37_test_fpaths["train_csv_path"]
    val_csv_path = ICC_5_train_37_test_fpaths["val_csv_path"]

    tokenizer = SpokenDialogueTokenizer(pretrained_model_name_or_path="gpt2")
    dm = TurnGPTFinetuneDM(
        tokenizer=tokenizer,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        conversation_id_key="convID",
        utterance_key="Utterance",
        save_dir=results_path,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    train_dataloader = dm.train_dataloader()

    for item in train_dataloader:
        print(item["input_ids"].tolist())
        for item in item["input_ids"]:
            decoded = tokenizer.decode(item.tolist())
            print(f"Decoded: {decoded}")
