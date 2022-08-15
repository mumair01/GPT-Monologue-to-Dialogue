# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 09:12:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 12:46:47


import sys
import os
from typing import Any, List

import glob
import shutil

def read_text(file_path : str):
    with open(file_path,"r") as f:
        return f.readlines()

def write_text(file_path : str, data : List[str], mode="w+"):
    with open(file_path, mode) as f:
        f.writelines(data)

def find_files_in_dir(dir_path : str, ext : str, recursive : bool = False) -> List[str]:
    return glob.glob(f"{dir_path}/*{ext}",recursive=recursive)

def get_filename(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]

def get_extension(file_path : str) -> str:
    return os.path.splitext(os.path.basename(file_path))[1][1:]

def create_dir(dir_path : str, overwrite : bool = False) -> None:
    if os.path.isdir(dir_path) and overwrite:
        shutil.rmtree(dir_path)
    os.makedirs(dir_path,exist_ok=True)


def remove_file(file_path : str):
    if os.path.isfile(file_path):
        os.remove(file_path)
