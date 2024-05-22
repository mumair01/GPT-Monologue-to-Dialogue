# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 08:58:15
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-22 07:46:34

import torch
import numpy as np

from typing import List, Dict, Any, Optional
from LLMDialogue.utils.general import get_system_device


class Model:
    pass


class LanguageModel(Model):
    """Abstract class the describes the methods required by a model"""

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        # Internal Attrs.
        self._model = None
        self._tokenizer = None
        self.seed = seed
        self.device = None

        # Set the seed.
        if self.seed is not None:
            np.random.seed(seed)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def load(self, *args, **kwargs):
        raise NotImplementedError()

    def finetune(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __repr__(self):
        return (
            f"Base model: {self._model}\n"
            f"Base tokenizer: {self._tokenizer}\n"
            f"device: {self.device}"
        )

    def to(
        self,
        device: Optional[torch.device] = None,
    ):
        assert self._model is not None
        if device is None:
            device = get_system_device()
        self.device = device
        self._model.to(device)

    def eval(self):
        self._model.eval()

    def encode(self, *args, **kwargs):
        raise NotImplementedError()
