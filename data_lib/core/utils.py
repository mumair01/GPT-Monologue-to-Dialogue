# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 09:12:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-01 02:52:24


import sys
import os
from typing import Any, List

import glob
import shutil

from typing import Callable

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

def process_file(file_path : str, process_fn : Callable) -> None:
    """
    Given a file at the specified path, run the given function with the file
    passed in
    """
    assert os.path.isfile(file_path) , \
        f"ERROR: {file_path} does not exist"

    return process_fn(file_path)


def process_files_in_dir(dir_path, file_ext : str, process_fn : Callable,
        recursive : bool = False):
    """
    Process all the files in the given directory with the given extension
    and using hre given process_fn

    """
    assert os.path.isdir(dir_path), \
        f"ERROR: {dir_path} does not exist"

    file_paths = find_files_in_dir(dir_path, file_ext,recursive)
    data = []
    pbar = tqdm(total=len(file_paths))
    for file_path in file_paths:
        pbar.set_description(desc=f"Applying {process_fn} to {get_filename(file_path)}")
        res = process_file(
            file_path=file_path,
            process_fn=process_fn
        )

        data.append(res)
        pbar.update()
    return data
