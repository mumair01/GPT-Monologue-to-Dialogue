# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-29 17:06:25
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 11:30:57


import sys
import os
from hydra_zen import hydrated_dataclass
from typing import List, Dict, Tuple, Any, Optional
from LLMDialogue.baseConfigs import BaseConfigs


@hydrated_dataclass
class WANDBTrackerConfigs:
    project: str = "gpt_monologue_to_dialogue"
    entity: str = "hilab"
    mode: str = "online"
    run_name: Optional[str] = None
