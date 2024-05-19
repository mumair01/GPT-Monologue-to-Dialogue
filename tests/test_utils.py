# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 15:39:59
# @Last Modified by:   Muhammad Umair

import sys
import os
import toml
import shutil

from typing import Dict, List, Dict, Any


# @Last Modified time: 2024-05-19 15:40:32
def reset_dir(dir_path: str):
    if os.path.isdir(dir_path) and dir_path != ".":
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def get_filename(file_path: str):
    if not os.path.isfile(file_path):
        raise Exception(f"Not a file: {file_path}")
    return os.path.splitext(file_path)[0]


def get_extension(file_path: str):
    if not os.path.isfile(file_path):
        raise Exception(f"Not a file: {file_path}")
    return os.path.splitext(file_path)[0][1:]


def get_dir_name(dir_path: str):
    if not os.path.isfile(dir_path):
        raise Exception(f"Not a directory: {dir_path}")
    return os.path.basename(dir_path)


def load_toml(file_path: str) -> Dict:
    if not os.path.isfile(file_path) and get_extension(file_path) != "toml":
        raise Exception(f"Unable to read file: {file_path}")
    return toml.load(file_path)


def load_text(file_path: str):
    if not os.path.isfile(file_path) and get_extension(file_path) != "txt":
        raise Exception(f"Unable to read file: {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()
        return lines


def prepend_to_str_dict(s: str, d: Dict) -> Dict:
    return {k: os.path.join(s, v) for k, v in d.items()}


def load_inference_text_from_file(inference_text_path: str) -> List:
    """
    Load text data from a text file and separate the paragraphs into lists of
    strings.
    """
    with open(inference_text_path, "r") as f:
        data = f.read().split("\n\n")
    data = [item.replace("\n", "") for item in data]
    return data
