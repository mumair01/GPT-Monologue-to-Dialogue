# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 19:38:21
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-19 19:38:23

# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-08-09 14:37:43
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-08-28 10:13:01


import sys
import os
import logging
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, asdict, is_dataclass
import json
from datetime import datetime

from .general import reset_dir

from typing import List, Dict, Any

# TODO: Add a helper for logging dictionaries in a readable format to help debug.


########################
# Logging  Private Helpers
########################


def _clear_handlers(logger: logging.Logger) -> None:
    """
    Remove all the handlers from the given logger.

    Parameters
    ----------
    logger : logging.Logger
    """
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


########################
# Logging config methods
########################


def create_stream_handler(
    logging_level: int | str = logging.INFO,
    log_format: str = "%(asctime)s | %(levelname)s | %(module)s | %(message)s",
) -> logging.Handler:
    """
    Create a handler for the output stream

    Parameters
    ----------
    logging_level : int | str, optional
        Level for the handler, by default logging.INFO
    log_format : str, optional
        Output format for the handler,
        by default "%(asctime)s | %(levelname)s | %(module)s %(funcName)s | %(message)s"

    Returns
    -------
    logging.Handler
    """
    formatter = logging.Formatter(log_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)
    return stream_handler


def create_file_handler(
    outdir: str = "./logs",
    fname: str = "process",
    logging_level: int | str = logging.INFO,
    log_format: str = "%(asctime)s | %(levelname)s | %(module)s %(funcName)s | %(message)s",
    reset_outdir: bool = False,
) -> logging.Handler:
    """
    Create a file handler.

    Parameters
    ----------
    outdir : str, optional
        Path to the output directory, by default "./logs"
    fname : str, optional
        Name of the log file (without extension), by default "process"
    logging_level : int | str, optional
        Level for the handler, by default logging.INFO
    log_format : str, optional
        Output format for the handler,
        by default "%(asctime)s | %(levelname)s | %(module)s %(funcName)s | %(message)s"
    reset_dir : bool, optional
        Reset the output directory if True,by default False
    Returns
    -------
    logging.Handler
    """
    # Create the log file path
    if reset_outdir and os.path.isdir(outdir):
        reset_dir(outdir)
    os.makedirs(outdir, exist_ok=True)
    now = datetime.now()
    dt = now.strftime("%d_%m_%Y__%H_%M_%S")
    log_fpath = f"{outdir}/{fname}_{dt}.log"
    # This overwrites existing files.
    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(filename=log_fpath, mode="w")
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(formatter)
    return file_handler


def configure_logger(
    logger: logging.Logger = None,
    root_logging_level: str | int = None,
    handlers: List[logging.Handler] = [],
    clear_handlers: bool = False,
) -> None:
    """
    Configure a basic logger, primarily for handling logging to stream and
    console.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger object. If None, uses the root logger, by default None
    root_logging_level : str | int, optional
        Level to be set for the logger supplied. If None, sets the minimum
        level from the handlers provided, by default None
    handlers : List[logging.Handler], optional
        Supply any additional handlers that should be added to the logger,
        by default []
    clear_handlers: bool, optional
        If True, clears all the handlers in the existing logger before
        adding the provided handlers, by default False
    """
    # If no logger is provided, configure the root
    logger = logger or logging.getLogger()

    # Clear existing handlers
    if clear_handlers:
        _clear_handlers(logger)
    # If not logging level is provided for root, then simply use the minimum
    # level from all handlers.
    if not root_logging_level:
        logger.setLevel(int(np.min([h.level for h in handlers])))
    else:
        logger.setLevel(root_logging_level)

    # Add the handlers
    for handler in handlers:
        logger.addHandler(handler)


########################
# Logging helper methods.
########################


def log_format(data: Any) -> str:
    """
    Format the input in a manner that is is easily viewable in a log file.

    Parameters
    ----------
    data : Any

    Returns
    -------
    str
        Pretty string representation for the input data
    """
    if isinstance(data, dict):
        return json.dumps(data, indent=4)
    elif isinstance(data, list):
        return ", ".join(data)
    elif is_dataclass(data):
        return log_format(asdict(data))
    else:
        return data
