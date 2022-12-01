# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 10:46:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-01 03:19:26

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

class ICCDataset:
    """
    Prepares the ICC dataset and can also act as a loader.
    """

    _VARIANTS = ("no_labels", "special_labels")
    _EXT = "cha"

    _CSV_HEADERS = ["convName","convID", "Utterance"]


    def __init__(self, dir_path : str):
        assert os.path.isdir(dir_path), \
            f"ERROR: Specified directory {dir_path} does not exist"
        self.dir_path = dir_path

    def __call__(self, variant : str, save_dir : str, outfile : str):
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
                item.insert(1,conv_no)
                combined.append(item)

        # Generate the save path and save
        create_dir(save_dir)
        partial_save_path = os.path.join(
            save_dir,f"{outfile}_{variant}")
        csv_path = f"{partial_save_path}.csv"
        self._save_dataset_as_csv(csv_path, combined)


    def read_dataset_csv(
        self,
        csv_path : str,
        start_conv_no : int = 0,
        end_conv_no : int = -1,
    ) -> List[pd.DataFrame]:
        """
        Read a csv file prepared by this Dataset and obtain conversations b/w
        [start_conv_no, end_conv_no].
        """
        df = self._load_dataset_from_csv(csv_path)
        if end_conv_no > len(conversation_dfs) or end_conv_no == -1:
            end_conv_no = len(conversation_dfs)
        assert len(conversation_dfs) >= end_conv_no
        assert start_conv_no < end_conv_no
        conversation_dfs = conversation_dfs[start_conv_no:end_conv_no]
        return conversation_dfs

    def _process_file(self, cha_path : str, variant : str):
        assert os.path.isfile(cha_path), f"ERROR: {cha_path} does not exist"

        conv_name = get_filename(cha_path)
        conv = read_text(cha_path)

        # Create and apply normalizer sequence
        normalizer_seq = create_normalizer_sequence(
            **self._get_variant_normalizer_params(variant))

        data = []
        for j in range(len(conv)):
            target_str = conv[j].strip()

            # Split on punctuation
            split_toks = re.split(r"\. |\?|\t+", target_str)
            # Apply normalizer
            split_toks = [normalizer_seq(toks) for toks in split_toks]

            split_toks = [tok for tok in split_toks if len(tok) > 0]
            if len(split_toks) > 0:
                split_toks = " ".join(split_toks)
                data.append([conv_name,split_toks])
        return data

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
                "remove_words" :  ["sp1", "sp2", "start", "end"],
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
                "replace_words" : {
                    "sp1" : "<SP1>",
                    "sp2" : "<SP2>",
                    "start" : "<START>",
                    "end" : "<END>"
                },
                "remove_words" :  [],
                "custom_regex" : "(\*)"
            }
        else:
            raise NotImplementedError()

    def _save_dataset_as_csv(self, csv_path, dataset):
        # Save the data as a dataframe
        dataset_df = pd.DataFrame(dataset, columns=self._CSV_HEADERS)
        # Generate the save path and save
        dataset_df.to_csv(csv_path)

    def _load_dataset_from_csv(self, csv_path):
        df = pd.read_csv(csv_path,names=self._CSV_HEADERS, index_col=0)
        # conversation_dfs = [df.loc[df[conv_key] == i] for i in range(
        # np.max(df[conv_key].unique()) + 1)]
        return df

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--path", type=str, required=True,
#         help="ICC .cha file path or directory containing .cha files")
#     parser.add_argument(
#         "--variant", type=str, help="Variant of the ICC to generate")
#     parser.add_argument(
#         "--outdir", type=str, default="./", help="Output directory")
#     parser.add_argument(
#         "--outfile", type=str,  help="Name of the output file")

#     args = parser.parse_args()


#     dataset = ICCDataset(args.path)
#     dataset(args.variant, args.outdir, args.outfile)
