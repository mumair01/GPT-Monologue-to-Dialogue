# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-11-27 11:39:52
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-11 21:05:48

import sys
import os
import toml
import shutil

from typing import Dict

# GLOBALS
ROOT_PATH = "/cluster/tufts/deruiterlab/mumair01/projects/gpt_monologue_dialogue"
CONFIG_TOML_PATH = os.path.join(ROOT_PATH,"tests/configs.toml")

def reset_dir(dir_path : str):
    if os.path.isdir(dir_path) and dir_path != ".":
        shutil.rmtree(dir_path)
    os.makedirs(dir_path,exist_ok=True)

def get_filename(file_path : str):
    if not os.path.isfile(file_path):
        raise Exception(f"Not a file: {file_path}")
    return os.path.splitext(file_path)[0]

def get_extension(file_path : str):
    if not os.path.isfile(file_path):
        raise Exception(f"Not a file: {file_path}")
    return os.path.splitext(file_path)[0][1:]

def get_dir_name(dir_path : str):
    if not os.path.isfile(dir_path):
        raise Exception(f"Not a directory: {dir_path}")
    return os.path.basename(dir_path)

def load_toml(file_path : str) -> Dict:
    if not os.path.isfile(file_path) and get_extension(file_path) != "toml":
        raise Exception(f"Unable to read file: {file_path}")
    return toml.load(file_path)

def load_text(file_path : str):
    if not os.path.isfile(file_path) and get_extension(file_path) != "txt":
        raise Exception(f"Unable to read file: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines

def load_configs() -> Dict:
    """Get the configs required for various aspects of testing"""
    return load_toml(CONFIG_TOML_PATH)
