# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-30 11:44:23
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 11:50:42


import sys
import os
from hydra_zen import hydrated_dataclass, MISSING
from typing import List, Dict, Tuple, Any, Optional
from LLMDialogue.baseConfigs import BaseConfigs


@hydrated_dataclass
class GPT2ModelConfig:
    load: Dict = {
        "model_checkpoint": "gpt2",
        "tokenizer_checkpoint": "gpt2",
        "tokenizer_pad_token": MISSING,
        "tokenizer_eos_token": MISSING,
        "tokenizer_additional_special_tokens": [],
    }
    finetune: Dict = {
        "num_train_epochs": 30,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 300,
        "logging_steps": 10,
        "logging_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "run_name": None,
    }


@hydrated_dataclass
class GPT2LabeledModelConfig(GPT2ModelConfig):
    load: Dict = {
        # Update load options with specific tokens for labeled models
        "tokenizer_pad_token": "<PAD>",
        "tokenizer_eos_token": "<|endoftext|>",
        "tokenizer_additional_special_tokens": ["<SP1>", "<SP2>", "<START>", "<END>"],
    }


@hydrated_dataclass
class TurnGPTLMModelConfig:
    load: Dict = {
        "pretrained_model_name_or_path": "gpt2",
        "model_head": "LMHead",
        "load_pretrained_configs": True,
        "omit_speaker_ids": False,
        "no_train_first_n": 5,
        "learning_rate": 1e-4,
        "tokenizer_additional_special_tokens": [],
    }
    finetune: Dict = {
        "max_epochs": 30,
        "log_every_n_steps": 1,
        "save_top_k": 1,
        # Optional fields for future use (uncomment if needed)
        # "report_to": Optional[str],
    }
