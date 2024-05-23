# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 09:05:51
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 09:10:18


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
from LLMDialogue.utils.MLTrackers import WANDBTracker, MLTracker

import logging

logger = logging.getLogger(__name__)

_TRACKERS = {
    "wandb": WANDBTracker,
}


def run_pipeline_with_tracker(
    tracker_type: str,
    pipe: PipelineBuilder,
    **tracker_kwargs,
) -> MLTracker:
    """
    Utility method that loads the specified type of tracker with the specified
    arguments and run the associated pipeline.
    """
    # Create the tracker
    assert (
        tracker_type in _TRACKERS
    ), f"Unsupported tracker type, choose from: {list(_TRACKERS.keys())}"

    tracker = _TRACKERS[tracker_type]()
    tracker.start_run(**tracker_kwargs)
    pipe.run()
    tracker.finish_run()
    return tracker
