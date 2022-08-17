# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 10:36:32
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 11:59:15

import sys
import os
from tqdm import tqdm


from typing import Callable

from data_lib.utils import find_files_in_dir, get_filename



def process_file(file_path : str, process_fn : Callable) -> None:
    assert os.path.isfile(file_path) , \
        f"ERROR: {file_path} does not exist"

    return process_fn(file_path)


def process_files_in_dir(dir_path, file_ext : str, process_fn : Callable,
        recursive : bool = False):
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

