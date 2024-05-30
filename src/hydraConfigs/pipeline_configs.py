# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-30 11:50:49
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 11:53:04

import sys
import os
from hydra_zen import hydrated_dataclass, MISSING
from typing import List, Dict, Tuple, Any, Optional
from LLMDialogue.baseConfigs import BaseConfigs


@hydrated_dataclass
class CPPipelineConfigs:
    N: int = -1
    context_buffer_size: int = 512
    probability_method: str = "eos"
