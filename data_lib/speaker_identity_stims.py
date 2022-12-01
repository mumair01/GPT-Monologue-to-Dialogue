# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 12:50:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-01 03:19:39

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

from data_lib.core import (
    read_text,
    get_extension,
    get_filename,
    create_normalizer_sequence,
    process_files_in_dir,
    create_dir,
    remove_file
)


class SpeakerIdentityStimuliDataset:
    """
    Prepares the SpeakerIdentityStimulus dataset and can also act as a loader.
    """

    _VARIANTS = ("no_labels", "special_labels")
    _EXT = "cha"

    def __init__(self, dir_path : str):
        assert os.path.isdir(dir_path), \
            f"ERROR: Specified directory {dir_path} does not exist"
        self.dir_path = dir_path

    def __call__(self, variant : str, save_dir : str):
        assert variant in self._VARIANTS, \
            f"ERROR: Specified variant not defined: {variant}"

        res = process_files_in_dir(
            dir_path=self.dir_path,
            file_ext=self._EXT,
            process_fn=partial(self._process_file,variant=variant),
            recursive=False
        )

        # Add conversation number to each conv.
        combined = []
        for conv_no, conv_data in enumerate(res):
            for item in conv_data:
                item.insert(0,conv_no)
                combined.append(item)

        # Generate the save path and save
        create_dir(save_dir)
        partial_save_path = os.path.join(save_dir,f"speaker_identity_stimuli_{variant}")
        csv_path = f"{partial_save_path}.csv"
        remove_file(csv_path)
        self._save_dataset_as_csv(csv_path, combined)


    def _process_file(self, cha_path : str, variant : str):
        assert os.path.isfile(cha_path),  f"ERROR: {cha_path} does not exist"

        conv_name = get_filename(cha_path)
        conv = read_text(cha_path)

        # Ignore / Remove all lines that start with comment marker.
        conv = [line for line in conv if line[0] != "@"]

        # Create and apply normalizer sequence
        normalizer_seq = create_normalizer_sequence(
            **self._get_variant_normalizer_params(variant))
        conv = [normalizer_seq(toks) for line in conv \
            for toks in re.split(r"\. |\?|\t+|:", line) ]
        conv = [line for line in conv if len(line) > 0]

        # Assumption: There are only two turns in the conversation.
        if variant == "no_labels":
            if conv[0] == conv[2]:
                conv = [f"{conv[1].strip()} {conv[3].strip()}"]
            else:
                conv = [f"{conv[1].strip()}",  f"{conv[3].strip()}"]
        elif variant == "special_labels":
            speakers = [line for line in conv if re.match(r"(sp)[0-9]",line)]
            lines = [line for line in conv if not re.match(r"(sp)[0-9]",line)]
            conv = []
            for sp, line in zip(speakers, lines):
                sp = f"<SP{sp[-1]}>"
                conv.append(f"{sp} {line.strip()} {sp}")
            assert len(conv) == 2
            conv.insert(0,"<START>")
            conv.append("<END>")

        # Add the conversation name to each item in list
        conv = [[conv_name, turn_no, item] for turn_no,item in enumerate(conv)]
        return conv

    def _get_variant_normalizer_params(self, variant : str):
        if variant == "no_labels":
            return {
                "lowercase" : True,
                "unicode_normalizer" : "nfd",
                "strip_accents" : True,
                "remove_punctuation" : True,
                "remove_extra_whitespaces" : True,
                "add_whitespace_punc" : True,
                "replace_words" : {},
                "remove_words" :  ["start", "end"],
                "custom_regex" : "(\*)"
            }
        elif variant == "special_labels":
            return {
                "lowercase" : True,
                "unicode_normalizer" : "nfd",
                "strip_accents" : True,
                "remove_punctuation" : True,
                "remove_extra_whitespaces" : True,
                "add_whitespace_punc" : True,
                "replace_words" : {},
                "remove_words" :  [],
                "custom_regex" : "(\*)"
            }
        else:
            raise NotImplementedError()

    def _save_dataset_as_csv(self, csv_path, dataset):
        # Save the data as a dataframe after sorting.
        dataset_df = pd.DataFrame(
            dataset,
            columns=["convID","convName","turnNo", "Utterance"]
        )
        dataset_df.sort_values(
                by=["convID","turnNo"],ignore_index=True, inplace=True)
        dataset_df = dataset_df.drop(columns=["turnNo"])
        dataset_df.to_csv(csv_path)

    def _load_dataset_from_csv(self, csv_path):
        df = pd.read_csv(csv_path,names=self._CSV_HEADERS, index_col=0)
        # conversation_dfs = [df.loc[df[conv_key] == i] for i in range(
        # np.max(df[conv_key].unique()) + 1)]
        return df

# if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--path", type=str, required=True,
    #     help="ICC .cha file path or directory containing .cha files")
    # parser.add_argument(
    #     "--variant", type=str, help="Variant of the ICC to generate")
    # parser.add_argument(
    #     "--outdir", type=str, default="./", help="Output directory")

    # args = parser.parse_args()

    # dataset = SpeakerIdentityStimuliDataset(args.path)
    # dataset(args.variant, args.outdir)