# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 12:50:18
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-15 17:05:41

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

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

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

    @property
    def special_labels_variant_labels(self):
        return {
            "speaker_base" : "<SP{}>",
            "start" : "<START>",
            "end" : "<END>"
        }

    def _process_file(self, cha_path : str, variant : str):
        assert os.path.isfile(cha_path),  f"ERROR: {cha_path} does not exist"

        if variant == "no_labels":
            return self._process_no_labels_variant(cha_path)
        elif variant == "special_labels":
            return self._process_special_labels_variant(cha_path)
        else:
            raise NotImplementedError(
                f"ERROR: Variant is not defined: {variant}"
            )

    # TODO: Again, similar to the ICC dataset, this combines the turns by the same
    # speaker into a single string, which might influence the probabilities
    # of the second turn since it might not be syntactically coherent...
    # I should refer back to the TurnGPT paper and see how they did this...
    def _process_no_labels_variant(self, cha_path):
        """
        Removes the headers speaker labels from the turns in the given file
        and combines turns from the same speaker into a single string - since
        this is needed by TurnGPT tokenizer.
        """
        conv_name = get_filename(cha_path)
        conv = read_text(cha_path)

        # Ignore / Remove all lines that start with comment marker.
        conv = [line for line in conv if line[0] != "@"]

        # Create and apply normalizer sequence.
        normalizer_seq = create_normalizer_sequence(
            remove_words = ["start", "end"],
            custom_regex = "(\*)"
        )
        conv = [normalizer_seq(toks) for line in conv \
            for toks in re.split(r"\. |\?|\t+|:", line) ]
        conv = [line for line in conv if len(line) > 0]

        # Assumption: There are only two turns in the conversation
        assert len(conv) == 4 # 4 here since we split the speakers.
        # NOTE: IMPORTANT: Using the <ts> label to merge but this
        # should be the same as the label that is used by the tokenizer
        # later.
        if conv[0] == conv[2]:
            conv = [f"{conv[1].strip()}<ts> {conv[3].strip()}<ts>"]
        else:
            conv = [f"{conv[1].strip()}<ts>",  f"{conv[3].strip()}<ts>"]

        # Add the conversation name to each item in list
        conv = [[conv_name, turn_no, item] for turn_no,item in enumerate(conv)]
        return conv

    def _process_special_labels_variant(self, cha_path):
        """
        Removes the headers from the given input files, wraps each sequence
        in a start and end token and each turn with the speaker labels of the
        given speaker. These speaker labels act as explicit indications of a new
        speaker.
        NOTE: The processing model should use the speaker labels here in it's
        processing. This has now been added as a property of the dataset.
        """
        conv_name = get_filename(cha_path)
        conv = read_text(cha_path)

        # Ignore / Remove all lines that start with comment marker.
        conv = [line for line in conv if line[0] != "@"]

        # Create and apply normalizer sequence.
        normalizer_seq = create_normalizer_sequence(
            custom_regex = "(\*)"
        )
        conv = [normalizer_seq(toks) for line in conv \
            for toks in re.split(r"\. |\?|\t+|:", line) ]
        conv = [line for line in conv if len(line) > 0]

        speakers = [line for line in conv if re.match(r"(sp)[0-9]",line)]
        lines = [line for line in conv if not re.match(r"(sp)[0-9]",line)]
        conv = []
        for sp, line in zip(speakers, lines):
            sp = self.special_labels_variant_labels["speaker_base"].format(sp[-1])
            conv.append(f"{sp} {line.strip()} {sp}")
        assert len(conv) == 2
        conv.insert(0,self.special_labels_variant_labels["start"])
        conv.append(self.special_labels_variant_labels["end"])

        # Add the conversation name to each item in list
        conv = [[conv_name, turn_no, item] for turn_no,item in enumerate(conv)]
        return conv

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
        return pd.read_csv(csv_path,names=self._CSV_HEADERS, index_col=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help=".cha file path or directory containing .cha files")
    parser.add_argument(
        "--variant", type=str, help="Variant of the ICC to generate")
    parser.add_argument(
        "--outdir", type=str, default="./", help="Output directory")

    args = parser.parse_args()

    dataset = SpeakerIdentityStimuliDataset(dir_path=args.path)
    dataset(
        variant=args.variant,
        save_dir=args.outdir
    )