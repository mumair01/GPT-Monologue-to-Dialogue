# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-21 15:49:31
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-23 12:54:23

import sys
import os
import numpy as np
import torch
import hashlib
import pytorch_lightning as pl
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

from LLMDialogue.baseConfigs import BaseConfigs
from LLMDialogue.models import LanguageModel, Model
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


# TODO: Eventually make a base config and builder class that can be used for
# both datasets and pipelines.


class PipelineConfig:

    def __init__(
        self,
        name: str = "unknown",
        variant_name: str = "default",
        version: str = "0.0.0",
        data_dir: str = None,
        data_files: Optional[Dict] = None,
        cache_dir: str = None,
        clear_cache: bool = False,
        compress_archive: bool = True,
        seed: Optional[int] = 42,
        verbose: bool = False,
        device: torch.device = get_system_device(),
        **config_kwargs,
    ):
        now = datetime.now()
        self.date_created = now.strftime("%d_%m_%Y__%H_%M_%S")
        self.name = name
        self.variant_name = variant_name
        self.version = version
        self.data_dir = data_dir
        self.data_files = data_files
        self.cache_dir = cache_dir
        self.clear_cache = clear_cache
        self.config_id = self.create_config_id(config_kwargs)
        if cache_dir is None:
            self.cache_dir = f"{BaseConfigs.pipeline.cachePath}/{self.config_id}"
        else:
            self.cache_dir = cache_dir
        self.archive_dir = os.path.join(self.cache_dir, "archive")
        self.compress_archive = compress_archive
        self.seed = seed
        self.verbose = verbose
        self.device = device
        # Set all the attrs from a dictionary
        for k, v in config_kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return log_format(self.to_dict())

    def update(self, k: str, v: Any) -> None:
        setattr(self, k, v)

    def pop(self, k: str) -> None:
        delattr(self, k)

    def copy(self) -> "PipelineConfig":
        raise NotImplementedError()

    def to_dict(self) -> Dict:
        # NOTE: Should be sanitized by the sub-class!
        return self.__dict__

    def create_config_id(self, config_kwargs: Dict) -> str:
        # Create a unique string based on the key value pairs and create a
        # unique hash
        s = ",".join([self.name, self.variant_name, self.version])
        if self.data_dir is not None:
            s += f",{get_dirname(self.data_dir)}"

        config_kwargs_k = ""
        for k, v in config_kwargs.items():
            config_kwargs_k += f",{str(k)}:{str(v)}"
        # Combine the strings
        s += config_kwargs_k
        hashed = hashlib.sha256(bytes(s, "utf-8")).hexdigest()
        return hashed


class PipelineInfo:
    def __init__(self, data: Dict, **kwargs):
        self.data = data
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return log_format(self.kwargs)

    def to_dict(self):
        return self.__dict__


class PipelineBuilder:
    _CONF_FNAME = "conf"
    _DATA_FNAME = "results"

    def __init__(
        self,
        config: PipelineConfig | None,
    ):
        self.data = {}  # Empty unless changed by sub-class.
        self.config = config
        # Use the config to build the config dir.
        if config is not None:
            if self.config.clear_cache:
                remove_dir(self.config.cache_dir)
            os.makedirs(self.config.cache_dir, exist_ok=True)
            os.makedirs(self.config.archive_dir, exist_ok=True)
            assert os.path.isdir(self.config.archive_dir)
        # Set a seed if there is one.
        if config.seed:
            np.random.seed(config.seed)

    def run(self) -> None:
        raise NotImplementedError()

    def info(self) -> PipelineInfo:
        return PipelineInfo(data=deepcopy(self.data), configs=self.config.to_dict())

    def load_from_disk(self) -> Any:
        raise NotImplementedError()

    def _load_data_from_disk(
        self,
        fname: str = _DATA_FNAME,
    ) -> Dict:
        assert os.path.isdir(
            self.config.cache_dir
        ), f"Pipeline cache dir does not exist: {self.config.cache_dir}"
        # conf_path = f"{cache_dir}/{self._CONF_FNAME}.json"
        data_path = f"{self.config.cache_dir}/{fname}.json"
        # config = conf_clazz(**read_json(conf_path))
        data = read_json(data_path)
        return data

    def __str__(self) -> str:
        return self.config.__str__()

    def _save_to_disk(self, config: PipelineConfig, data: Dict) -> None:

        assert os.path.exists(config.archive_dir)
        conf_path = f"{config.cache_dir}/{self._CONF_FNAME}.json"
        data_path = f"{config.cache_dir}/{self._DATA_FNAME}.json"
        tar_fpath = f"{config.cache_dir}/archive.tar"
        write_json(conf_path, config.to_dict())
        write_json(data_path, data)
        # Compress the compress dir
        if self.config.compress_archive:
            if os.listdir(config.archive_dir):
                make_tarfile(tar_fpath, config.archive_dir)
            # Remove everything in the compress folder
            for p in get_files_in_dir(config.archive_dir, ["*"]) + get_subdirs_in_dir(
                config.archive_dir
            ):
                if os.path.isfile(p):
                    remove_file(p)
                else:
                    remove_dir(p)
            remove_dir(config.archive_dir)
