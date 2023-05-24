# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-24 09:58:57
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-24 10:16:48

import sys
import os
from glob import glob
import pandas as pd


DIR_PATH = "/Users/muhammadumair/Desktop/Inference"
OUTPUT_PATH = "/Users/muhammadumair/Desktop/output"

_REMOVE_WORDS = ("<SP1>", "<SP2>", "<START>", "<END>")
_REMOVE_COLUMNS = ("context", "word")

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # Load the csv
    csv_paths = glob(f"{DIR_PATH}/*.csv")
    for path in csv_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        # Remove all rows with _REMOVE_WORDS
        for word in _REMOVE_WORDS:
            df = df[df["word"].str.contains(word) == False]
        # Remove specific columns
        df = df.drop(columns=list(_REMOVE_COLUMNS))
        df.to_csv(f"{OUTPUT_PATH}/{filename}.csv")
