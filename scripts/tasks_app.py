# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-23 17:27:58
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 16:10:11
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Callable

# TODO: Need this to access the scripts.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from LLMDialogue.utils.logging import (
    configure_logger,
    create_stream_handler,
)
from LLMDialogue.tasks import CPInference_task, finetune_task


import logging

logger = logging.getLogger(__name__)


configure_logger(
    handlers=[
        create_stream_handler(logging_level=logging.INFO),
        # create_file_handler(
        #     outdir=LOGGING_DIR,
        #     logging_level=logging.DEBUG,
        #     reset_outdir=False,  # Keep logs across runs.
        # ),
    ]
)

_TASKS_MAP = {
    "CPInference": CPInference_task,
    "finetune": finetune_task,
}


@hydra.main(
    config_path="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/configs/hydra",
    version_base=None,
    config_name="config",
)
def main(cfg: DictConfig):
    assert cfg.task in _TASKS_MAP, f"Task {cfg.task} not in {list(_TASKS_MAP.keys())}"
    _TASKS_MAP[cfg.task](cfg)


if __name__ == "__main__":
    main()
