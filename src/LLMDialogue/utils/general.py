# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 09:12:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-24 11:59:11


import sys
import os
from copy import deepcopy
import pandas as pd
import toml
import glob
from pathlib import Path
import shutil
import json
from tqdm import tqdm
from dataclasses import dataclass, fields
from pathlib import Path
import tarfile
import torch
import numpy as np
from typing import Any, List, Dict, Union, Callable, Tuple

import logging


logger = logging.getLogger(__name__)


from typing import Callable


def read_toml(path: str) -> Dict:
    """Load configuration from .toml file."""
    assert get_extension(path) == "toml", f"ERROR: Detected extension not toml: {path}"
    return toml.load(path)


def read_text(file_path: str):
    with open(file_path, "r") as f:
        return f.readlines()


def write_text(file_path: str, data: List[str], mode="w+"):
    with open(file_path, mode) as f:
        f.writelines(data)


def find_files_in_dir(dir_path: str, ext: str, recursive: bool = False) -> List[str]:
    return glob.glob(f"{dir_path}/*{ext}", recursive=recursive)


def get_filename(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def get_extension(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[1][1:]


def create_dir(dir_path: str, overwrite: bool = False) -> None:
    if os.path.isdir(dir_path) and overwrite:
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def remove_file(file_path: str):
    if os.path.isfile(file_path):
        os.remove(file_path)


def process_file(file_path: str, process_fn: Callable) -> None:
    """
    Given a file at the specified path, run the given function with the file
    passed in
    """
    assert os.path.isfile(file_path), f"ERROR: {file_path} does not exist"

    return process_fn(file_path)


def process_files_in_dir(
    dir_path, file_ext: str, process_fn: Callable, recursive: bool = False
):
    """
    Process all the files in the given directory with the given extension
    and using hre given process_fn

    """
    assert os.path.isdir(dir_path), f"ERROR: {dir_path} does not exist"

    file_paths = find_files_in_dir(dir_path, file_ext, recursive)
    data = []
    pbar = tqdm(total=len(file_paths))
    for file_path in file_paths:
        pbar.set_description(desc=f"Applying {process_fn} to {get_filename(file_path)}")
        res = process_file(file_path=file_path, process_fn=process_fn)

        data.append(res)
        pbar.update()
    return data


def get_parent_dir(path: str) -> str:
    """
    Get the path to a parent dir.

    Parameters
    ----------
    path : str
        Path to the dir whose parent get.

    Returns
    -------
    str
        Path to the parent dir.
    """
    return os.path.dirname(os.path.abspath(path))


def get_extension(path: str) -> str:
    """
    Get the extension to a file.

    Parameters
    ----------
    path : str
        Path to a file

    Returns
    -------
    str
        Extension (without the dot)
    """
    assert os.path.isfile(path), f"ERROR: Not a file {path}"
    return os.path.splitext(os.path.basename(path))[1][1:]


def get_filename(path: str) -> str:
    """Get the name of a given file"""
    assert os.path.isfile(path), f"ERROR: Not a file {path}"
    return os.path.splitext(os.path.basename(path))[0]


def get_dirname(path: str) -> str:
    """Get the name of a given directory"""
    # TODO: Removed explicit check for whether the directory exists.
    # assert os.path.isdir(path), f"ERROR: Not a dir: {path}"
    return os.path.basename(path)


def read_text(filepath: str) -> List:
    """Read a text file"""
    assert os.path.isfile(filepath), f"ERROR: Not a file {filepath}"
    with open(filepath, "r") as f:
        return f.readlines()


def read_json(filepath: str) -> Dict:
    """Read a json file"""
    assert os.path.isfile(filepath), f"ERROR: Not a file {filepath}"
    with open(filepath, "r") as f:
        return json.load(f)


def write_json(filepath: str, data: Any, mode="w") -> None:
    """Write a yaml file with specified mode"""

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open(filepath, mode) as f:
        json.dump(data, f, default=np_encoder)


def get_files_in_dir(
    dir_path: str,
    exts: List[str],
    recursive: bool = False,
) -> List[str]:
    """Get the files in a directory with the given extension"""

    assert os.path.isdir(dir_path), f"ERROR: Not a directory: {dir_path}"
    paths = list()
    for ext in exts:
        paths.extend(
            [str(path.absolute()) for path in Path(dir_path).rglob(f"*.{ext}")]
            if recursive
            else glob.glob(f"{dir_path}/*.{ext}")
        )
    return paths


def get_subdirs_in_dir(dir_path: str, recursive: bool = False) -> List[str]:
    """Obtain the subdirectories in a directory"""
    return glob.glob(f"{dir_path}/**/", recursive=recursive)


def reset_dir(dir_path: str) -> None:
    """
    Remove the provided dir and make a new one.
    """
    remove_dir(dir_path)
    os.makedirs(dir_path)


def remove_dir(dir_path: str) -> None:
    """
    Remove the given dir
    """
    if dir_path == ".":
        raise Exception(f"ERROR: Cannot delete root dir")
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def remove_file(fpath: str) -> None:
    """
    Remove the given file
    """
    if os.path.isfile(fpath):
        os.remove(fpath)


def copy(src_path: str, dest_path: str) -> str:
    assert os.path.exists(src_path)
    shutil.copy(src_path, dest_path)
    name = (
        get_dirname(src_path) if os.path.isdir(src_path) else os.path.basename(src_path)
    )
    return f"{dest_path}/{name}"


def move(src_path: str, dest_path: str) -> str:
    """
    Move a file or folder from source to destination.
    """
    shutil.move(src_path, dest_path)

    name = (
        get_dirname(src_path) if os.path.isdir(src_path) else os.path.basename(src_path)
    )
    return f"{dest_path}/{name}"


def check_unique(lst: List) -> bool:
    # Ensure that all items in the list are unique.
    _, counts = np.unique(lst, return_counts=True)
    return all(counts == 1)


def relpath(data: Any, start: str) -> Any:
    assert os.path.exists(start)
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = relpath(v, start)
    if isinstance(data, list):
        return [relpath(item, start) for item in data]
    if isinstance(data, str) and (os.path.isfile(data) or os.path.isdir(data)):
        return os.path.relpath(data, start)
    return data


def join_path(data: Union[List, Dict, str], root_path: str) -> Union[List, Dict, str]:
    """
    Make all the paths in the given dict, list, or str absolute
    """

    if isinstance(data, dict):
        # print("dict", data)
        for k, v in data.items():
            data[k] = join_path(v, root_path)
    if isinstance(data, list):
        # print("list", data)
        return [join_path(item, root_path) for item in data]
    if isinstance(data, str):
        # print("str", data)
        path = os.path.join(root_path, data)
        # NOTE: Attempt to make this a path i.e, check that the newly
        # constructed path exists. If it does not, then use the original
        # string instead.
        return path if os.path.exists(path) else data

    return data


def prepend_path_dict(root_path: str, data_dict: Dict, keys: List[str]) -> None:
    """
    Prepend the path to the given keys in a dict.
    """
    for k in keys:
        item = data_dict[k]
        if isinstance(item, list):
            mod_item = [os.path.join(root_path, p) for p in item]
            data_dict[k] = mod_item
        elif isinstance(item, str):
            data_dict[k] = os.path.join(root_path, item)
        else:
            raise Exception()


def project_root_path() -> str:
    return Path(__file__).parent.parent.parent


def dataclass_fields(class_or_instance: dataclass) -> List[str]:
    return [field.name for field in fields(class_or_instance)]


def filter_from_dict_of_dicts(
    dict_dicts: Dict[str, Dict],
    filter_fn: Callable,
):
    return {k: [item for item in v if filter_fn(item)] for k, v in dict_dicts.items()}


def fields_from_dict_of_dicts(
    dict_dicts: Dict[str, Dict],
    fields: List[str],
) -> Dict[str, Dict]:
    return {
        k: [{key: value for key, value in item.items() if key in fields} for item in v]
        for k, v in dict_dicts.items()
    }


def merge_dicts(dict1: Dict, dict2) -> Dict:
    dict2.update(dict1)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def is_item(items: List, condition: Callable, get_idx=False) -> int | bool:
    # -1 if item does not exist.

    res = np.nonzero(list(map(condition, items)))[0]
    if len(res) == 0:
        return -1 if get_idx else False
    else:
        return res.item() if get_idx else True


def read_elan_tab_file(path: str, columns: List[str]) -> pd.DataFrame:
    """
    Read a tab delimited text file
    """
    data = read_text(path)
    data = [item.strip() for item in data]
    data = [item.replace("\t\t", "\t") for item in data]
    data = [item.split("\t") for item in data]
    # data = [v for item in data for v in item if len(v) > 0 ]
    df = pd.DataFrame(data, columns=columns)
    return df


def greater_than_t(arr: np.ndarray, t: float) -> Tuple:
    idxs = np.where(arr > t)[0]
    return arr[idxs], idxs


def lower_than_t(arr: np.ndarray, t: float) -> Tuple:
    idxs = np.where(arr <= t)[0]
    return arr[idxs], idxs


def between_intervals(arr: np.ndarray, t_low: float, t_high: float) -> Tuple:
    assert t_high > t_low
    idxs = np.where(np.logical_and(arr >= t_low, arr <= t_high))[0]
    return arr[idxs], idxs


# Check that MPS is available
def get_system_device() -> torch.device:
    """
    Determine the device object that is available.

    Returns
    -------
    torch.device
        Return the device object that is available.
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.info(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            logger.info(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_random_colors(N) -> List:
    """
    Helper function to generate N random Hex colors
    """
    colors_hex = []
    for _ in range(N):
        # Generate random RGB values between 0 and 255
        color_rgb = np.random.randint(0, 256, size=3)
        # Convert RGB to hexadecimal
        color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
        colors_hex.append(color_hex)
    return colors_hex


def remove_keys_from_dict(d: Dict, keys: List[str] | str) -> Dict:
    d_new = deepcopy(d)
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k in d_new:
            del d_new[k]
    return d_new
