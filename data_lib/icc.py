# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 10:46:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-10 14:07:44

import sys
import os

import pandas as pd
import re
import argparse

from typing import List
from functools import partial

import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from data_lib.core import (
    read_text,
    get_extension,
    get_filename,
    create_normalizer_sequence,
    replace_word_from_string,
    remove_words_from_string,
    process_files_in_dir,
    create_dir,
    remove_file
)

# TODO: For the no labels dataset, how do we indicate separate turns by the same
# speaker? We should be able to indicate this in TurnGPT since the turns
# may be syntactically incoherent, but might make sense as individual subsequent
# turns by the same speaker.
# Should this really matter since there is no temporal embeddings. It might
# because subsequent sequences by speakers may matter.
# I might want to change the no labels dataset to include a speaker tag
# that may be used by turngpt - but I'm not sure how to even approach this
# problem.
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
        conversation_dfs = self._load_dataset_from_csv(csv_path)
        if end_conv_no > len(conversation_dfs) or end_conv_no == -1:
            end_conv_no = len(conversation_dfs)
        assert len(conversation_dfs) >= end_conv_no
        assert start_conv_no < end_conv_no
        conversation_dfs = conversation_dfs[start_conv_no:end_conv_no]
        return conversation_dfs

    @property
    def special_labels_variant_labels(self):
        return {
            "speaker_base" : "<SP{}>",
            "start" : "<START>",
            "end" : "<END>"
        }

    def _process_file(self, cha_path : str, variant : str):
        assert os.path.isfile(cha_path), f"ERROR: {cha_path} does not exist"

        if variant == "no_labels":
            return self._process_no_labels_variant(cha_path)
        elif variant == "special_labels":
            return self._process_labels_variant(cha_path)
        else:
            raise NotImplementedError(
                f"ERROR: Variant is not defined: {variant}"
            )


    def _process_no_labels_variant(self, cha_path):
        """
        Here, we want to remove all explicit speaker labels and make the assumption
        that speakers alternate for each utterance in the list.
        """
        conv_name = get_filename(cha_path)
        conv = read_text(cha_path)

        normalizer_seq = create_normalizer_sequence(
            remove_words=["start", "end"],
            custom_regex= "(\*)"
        )

        data = []
        for j in range(len(conv)):
            target_str = conv[j].strip()
            # Split on punctuation
            split_toks = re.split(r"\. |\?|\t+", target_str)
            split_toks = [normalizer_seq(tok) for tok in split_toks[:-1]]
            split_toks = [tok for tok in split_toks if len(tok) > 0]
            if len(split_toks) > 0:
                if len(data) > 0 and data[-1][-1][0] == split_toks[0]:
                    data[-1][-1][-1] += " " +  " ".join(split_toks[1:])
                elif len(split_toks) == 2:
                    data.append([conv_name, split_toks])
        # Removing the speaker labels
        for i in range(len(data)):
            conv_name, split_toks = data[i]
            data[i] = [conv_name, split_toks[1]]

        return data

    def _process_labels_variant(self, cha_path):
        """
        Here, we extract the text from the cha files and create a List
        of normalizer utterances from different speakers with explicit sequence
        start and end tokens, as well as explicit speaker tokens.
        """
        conv_name = get_filename(cha_path)
        conv = read_text(cha_path)

        # Using default normalizer for the text and replace normalizer for the
        # start, end, and speaker labels.
        text_normalizer_seq = create_normalizer_sequence()
        labels_normalizer_seq = create_normalizer_sequence(
            replace_words= {
                "sp1" : self.special_labels_variant_labels["speaker_base"].format("1"),
                "sp2" : self.special_labels_variant_labels["speaker_base"].format("2"),
                "start" : self.special_labels_variant_labels["start"],
                "end" : self.special_labels_variant_labels["end"]
            },
            custom_regex= "\*"
        )

        data = []
        for j in range(len(conv)):
            target_str = conv[j].strip()

            # Split on punctuation
            split_toks = re.split(r"\. |\?|\t+", target_str)

            # NOTE: split_toks is either len 1 or 3.
            # We only want to apply the normalizer to the actual text.
            if len(split_toks) == 3:
                split_toks[0] = labels_normalizer_seq(split_toks[0])
                split_toks[1] = text_normalizer_seq(split_toks[1])
                split_toks[2] = labels_normalizer_seq(split_toks[2])
            else:
                split_toks = [labels_normalizer_seq(tok) for tok in split_toks]

            split_toks = [tok for tok in split_toks if len(tok) > 0]
            if len(split_toks) > 0:
                split_toks = " ".join(split_toks)
                data.append([conv_name,split_toks])

        return data

    def _save_dataset_as_csv(self, csv_path, dataset):
        # Save the data as a dataframe
        dataset_df = pd.DataFrame(dataset, columns=self._CSV_HEADERS)
        # Generate the save path and save
        dataset_df.to_csv(csv_path)

    def _load_dataset_from_csv(self, csv_path):
        return pd.read_csv(csv_path,names=self._CSV_HEADERS, index_col=0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="ICC .cha file path or directory containing .cha files")
    parser.add_argument(
        "--variant", type=str, help="Variant of the ICC to generate")
    parser.add_argument(
        "--outdir", type=str, default="./", help="Output directory")
    parser.add_argument(
        "--outfile", type=str,  help="Name of the output file")

    args = parser.parse_args()

    dataset = ICCDataset(dir_path=args.path)
    dataset(
        variant=args.variant,
        save_dir=args.outdir,
        outfile=args.outfile
    )