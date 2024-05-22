# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-22 09:08:08
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-22 09:23:42

import sys
import os
import numpy as np
import torch
import hashlib
import pytorch_lightning as pl
import itertools
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Any, Callable, Optional, Tuple

from LLMDialogue.baseConfigs import BaseConfigs
from LLMDialogue.models import LanguageModel, Model
from LLMDialogue.datasets import DatasetBuilder, DatasetConfig, load
from LLMDialogue.pipelines.base import PipelineBuilder, PipelineConfig
from LLMDialogue.utils.general import (
    get_dirname,
    remove_dir,
    write_json,
    remove_file,
    make_tarfile,
    get_files_in_dir,
    get_subdirs_in_dir,
    read_json,
    get_system_device,
)
from LLMDialogue.utils.logging import log_format

import logging

logger = logging.getLogger(__name__)

_PIPELINE_NAME = "FinetunePipeline"


class LLMFinetunePipelineConfig(PipelineConfig):

    def __init__(
        self,
        model: LanguageModel,
        dm: pl.LightningDataModule | Any,
        finetune_kwargs: Dict,
        **kwargs,
    ):
        config_kwargs = {
            "name": _PIPELINE_NAME,
            "model": model,
            "dm": dm,
            "finetune_kwargs": finetune_kwargs,
        }
        # Remove a key if it exists in kwargs.
        config_keys = list(config_kwargs.keys())
        for k in config_keys:
            if k in kwargs:
                del config_kwargs[k]
        # Initialize the superclass.
        super().__init__(**kwargs, **config_kwargs)

    def to_dict(self) -> Dict:
        # Here, make sure we remove items that we are not serializable.
        d = self.__dict__
        d["model"] = str(type(d["model"]))
        d["dm"] = str(type(d["dm"]))
        return d


class LLMFinetunePipeline(PipelineBuilder):

    def __init__(self, config: Optional[LLMFinetunePipelineConfig]):
        super().__init__(config)

    def save_to_disk(self) -> None:
        assert os.path.isdir(self.config.archive_dir)
        super()._save_to_disk(deepcopy(self.config), self.data)

    def __str__(self) -> str:
        return self.config.__str__()

    ###########
    # REQUIRED PUBLIC METHODS
    ###########

    def run(self):
        # Save the data attribute
        self._finetune()

    ###########
    # REQUIRED PRIVATE METHODS
    ###########

    def load_from_disk(self) -> None:
        # Load the configs and the actual data form the data file and the
        # info files.
        # TODO: We can't do this yet because we can't serialize the model
        # and the dm to save them.
        raise NotImplementedError()

    ###########
    # Pipeline SPECIFIC PRIVATE METHODS
    ###########

    def _finetune(self):
        # Make sure that we process that data module - it must implement
        # a fit stage with a train and val dataloader.
        dm = self.config.dm
        model = self.config.model
        dm.prepare_data()
        dm.setup(stage="fit")
        # Now, move the model to the right device
        model.to(self.config.device)
        # Start finetuning!
        # NOTE: The config MUST contain the finetuning params for the specific
        # model that is provided!
        model.finetune(
            # DM must be supplied.
            dm=dm,
            **self.config.finetune_kwargs,
        )
