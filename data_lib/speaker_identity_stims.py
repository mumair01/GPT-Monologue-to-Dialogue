# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 12:50:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 13:25:39

# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 10:46:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 13:01:18

import sys
import os

import pandas as pd
import re
import argparse

from typing import List
from functools import partial

import shutil

from data_lib import (
    read_text,
    get_extension,
    get_filename,
    create_normalizer_sequence,
    process_files_in_dir,
    create_dir,
    remove_file
)


def preprocess_speaker_identity_stims(cha_path : str, variant : str) -> List:
    assert os.path.isfile(cha_path), \
        f"ERROR: {cha_path} does not exist"
    conv_name = get_filename(cha_path)
    conv = read_text(cha_path)

    # Ignore / Remove all lines that start with comment marker.
    conv = [line for line in conv if line[0] != "@"]

    # Build params based on data variant
    if variant == "monologue_gpt":
        remove_words = []
        replace_words = {}
    elif variant == "turngpt":
        replace_words = {}
        remove_words = ["sp1", "sp2", "start", "end"]
    else:
        raise NotImplementedError(
            f"ERROR: ICC data variant {variant} undefined"
        )
    # Create normalizer sequence
    normalizer_seq = create_normalizer_sequence(
        lowercase=True,
        unicode_normalizer="nfd",
        strip_accents=True,
        remove_punctuation=True,
        remove_extra_whitespaces=True,
        add_whitespace_punc=True,
        replace_words=replace_words,
        remove_words=remove_words,
        custom_regex = "(\*)"
    )

    data = []

    for j in range(len(conv)):
        target_str = conv[j].strip()

        # Split on punctuation
        split_toks = re.split(r"\. |\?|\t+|:", target_str)
        # Apply normalizer
        split_toks = [normalizer_seq(toks) for toks in split_toks]

        # Add speaker token to end for monologue gpt
        if variant == "monologue_gpt":
            # Removing existing speaker tokens to add the ones needed by the model.
            # NOTE: Assuming that speaker ids start from 1.
            for speaker_id in range(1,3):
                SPEAKER_TOK = "<SP{}>"
                split_toks = [SPEAKER_TOK.format(speaker_id) \
                    if re.match("sp{}".format(speaker_id),tok) else tok for tok in split_toks]
            split_toks.append(split_toks[0])

        split_toks = [tok for tok in split_toks if len(tok) > 0]

        if len(split_toks) > 0:
            split_toks = " ".join(split_toks)
            data.append([conv_name,split_toks])
    if variant == "monologue_gpt":
        data.insert(0,[conv_name,"<START>"])
        data.append([conv_name, "<END>"])
    return data


def process_speaker_identity_stims(path : str, variant : str, ext : str, outfile : str, out_dir : str):
    if os.path.isfile(path) and get_extension(path) == ext:
        res = preprocess_speaker_identity_stims(path,variant)
    else:
        res = process_files_in_dir(
            dir_path=path,
            file_ext=ext,
            process_fn=partial(preprocess_speaker_identity_stims,variant=variant),
            recursive=False
        )
    # Add conversation number to each conv.
    combined = []
    for conv_no, conv_data in enumerate(res):
        for item in conv_data:
            item.insert(1,conv_no)
            combined.append(item)
    # Save the data as a dataframe
    dataset_df = pd.DataFrame(combined, columns=["convName","convID", "Utterance"])
    # Generate the save path and save
    create_dir(out_dir)
    partial_save_path = os.path.join(out_dir,outfile)
    csv_path = f"{partial_save_path}.csv"
    remove_file(csv_path)
    dataset_df.to_csv(csv_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="ICC .cha file path or directory containing .cha files")
    parser.add_argument(
        "--variant", type=str, help="Variant of the ICC to generate")
    parser.add_argument(
        "--outfile", type=str, required=True,
        help="name of the output file")
    parser.add_argument(
        "--outdir", type=str, default="./", help="Output directory")

    args = parser.parse_args()

    process_speaker_identity_stims(
        path=args.path,
        variant=args.variant,
        outfile=args.outfile,
        out_dir=args.outdir,
        ext="cha"
    )
