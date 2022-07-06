# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-06 15:31:31
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-06 15:47:03


import sys
import os
import argparse
import random
import shutil
import pprint
from dataclasses import dataclass
from datetime import datetime
import yaml

import pandas as pd
import numpy as np
import torch
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


import logging
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class Configs:
    pass


def parse_configs(configs_data):
    pprint.pprint(configs_data)
    # Obtaining timestamp for output.
    now = datetime.now()
    ts = now.strftime('%m-%d-%Y_%H-%M-%S')
    # Creating configs
    configs = Configs()
    return configs

def save_configs(configs_data, file_path):
    with open(file_path,'w') as f:
        yaml.dump(configs_data,f)

def config_env(config_path):
    # Parse configs
    logger.info("Loading configurations from path: {}".format(config_path))
    with open(config_path,"r") as f:
        configs_data = yaml.safe_load(f)
        configs = parse_configs(configs_data)
    # Set seed if required
    if configs.env.seed != None:
        np.random.seed(configs.env.seed)
        torch.manual_seed(configs.env.seed)
    # Check input files exist
    assert os.path.isfile(configs.dataset.train_path)
    assert os.path.isfile(configs.dataset.val_path)
    # Create output directories
    os.makedirs(configs.results.save_dir)
    os.makedirs(configs.results.reports_dir,exist_ok=True)
    # Save the configs data to output dir.
    save_configs(configs_data, "{}/configs.yaml".format(configs.results.save_dir))
    return configs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",type=str,required=True, help="Configuration file path")
    args = parser.parse_args()
    # Load the configuration file and parse it
    configs = config_env(args.config)
    # -- Create loggers for this script
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(configs.results.reports_dir,"finetuning.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Starting finetuning.
    finetune(configs)