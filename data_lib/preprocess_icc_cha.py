# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-11 12:39:22
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-11 12:52:40
#######
# This script is inspired by 4.0-MU-GPT-ICC-Preprocess to convert .cha files
# of the ICC prepared by Julia to .csv and .txt files that can be used for
# finetuning.
#
######

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import json
from copy import deepcopy
from sklearn.utils import shuffle
import re
import glob

import random
import shutil
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# -------- GLOBALS.

# Define the start and end tokens
SPEAKER_TOK = "<SP{}>"
CONV_START_TOK = "<START>"
CONV_END_TOK = "<END>"

GLOBAL_SEED = 42

# -------- MAIN METHODS


def preprocess_huggingface_icc(cha_paths, seed=GLOBAL_SEED):
    """
    Creates a dataset dataframe from Julia's processed .cha files.
    Assumptions about the previous data:
        1. Has been previous pre-processed by Julia.
        2. Contains start / end tokens at the start and end of each .cha file.
        3. Each line format is SPX\t <text> \tSPX
        4. There are only two speakers per conversation.
    """
    cha_paths = deepcopy(cha_paths)
    cha_paths = shuffle(cha_paths,random_state=seed)
    pbar = tqdm(desc="Preprocessing ICC conversations", total=len(cha_paths))
    data = []
    for i in range(len(cha_paths)):
        with open(cha_paths[i],'r') as f:
            # Read all lines as a list
            conv_name = os.path.splitext(os.path.basename(cha_paths[i]))[0]
            conv = f.readlines()
            for j in range(len(conv)):
                target_str = conv[j].strip()
                split_toks = re.split(r"\. |\?|\t+", target_str)
                split_toks = [tok for tok in split_toks if len(tok) > 0]
                # Remove all punctuation and lowercase all
                split_toks = [re.sub(r'[^\w\s]', '', tok).lower() for tok in split_toks]
                # Remove any double whitespaces
                split_toks = [re.sub(' +', ' ', tok).lower() for tok in split_toks]
                # Removing existing speaker tokens to add the ones needed by the model.
                split_toks = [SPEAKER_TOK.format("1") if re.match(r"(sp1)", tok) else tok for tok in split_toks]
                split_toks = [SPEAKER_TOK.format("2") if re.match(r"(sp2)", tok) else tok for tok in split_toks]
                split_toks = [CONV_START_TOK if re.match('start',tok)  else tok for tok in split_toks]
                split_toks = [CONV_END_TOK if re.match('end',tok) else tok for tok in split_toks]
                if len(split_toks) == 3:
                    split_toks = [" ".join(split_toks)]
                    # split_toks = list( " ".join(split_toks))
                data.extend([(conv_name,i, tok) for tok in split_toks])
        pbar.update()
    dataset_df = pd.DataFrame(data, columns=["convName","convID", "Utterance"])
    return dataset_df

def main(args):
    assert os.path.isfile(args.path) or os.path.isdir(args.path), \
            "{} is not a valid path".format(args.path)

    if os.path.isfile(args.path) and \
            os.path.splitext(os.path.basename(args.path)[1]) == ".cha":
        cha_paths = [args.path]
    else:
        cha_paths = glob.glob("{}/*.cha".format(args.path))

    dataset_df = preprocess_huggingface_icc(cha_paths, seed=GLOBAL_SEED)
    dataset_df.to_csv(os.path.join(args.out_dir,args.outfile)+".csv")
    # Save the dataframe as a text file as well
    # NOTE: This is important to make sure that TextDataset can read these
    # files during finetuning.
    with open(os.path.join(args.out_dir,args.outfile)+".txt","w") as f:
        f.writelines("\n".join(dataset_df["Utterance"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="ICC .cha file path or directory containing .cha files")
    parser.add_argument(
        "--outfile", type=str, required=True,
        help="name of the output file")
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Output directory")

    args = parser.parse_args()
    main(args)