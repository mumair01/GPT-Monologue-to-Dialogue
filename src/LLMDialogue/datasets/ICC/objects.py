# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 19:48:35
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-21 09:19:12

import os
import sys
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
from LLMDialogue.datasets.ICC import ICCDataset


class ICCDatasetSplits:
    def __init__(
        self,
        data: Dict,
    ):
        self.data = data
        self.data_df_map = self._data_to_df(data)

    @property
    def split_names(self) -> List[str]:
        return list(self.data.keys())

    def to_csv(self, outdir: str) -> Dict[str, str]:
        os.makedirs(outdir, exist_ok=True)
        fpaths = dict()
        for split_name, split_df in self.data_df_map.items():
            outpath = f"{outdir}/{split_name}.csv"
            split_df.to_csv(outpath)
            fpaths[split_name] = outpath
        return fpaths

    def _data_to_df(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Knows how to convert the ICC data into a data frame for each split.
        """
        data_df_map = dict()
        for split_name, split_data in data.items():
            # We need to create list of lists for the data first.
            data_lists = list()
            for conv_name, conv_data in split_data.items():
                for i, utt in enumerate(conv_data["utterances"]):
                    # Store the conversation name, conversation ID, the turn number,
                    # and the actual utterance.
                    data_lists.append([conv_name, conv_data["id"], i, utt])
            df = pd.DataFrame(
                data_lists,
                columns=[
                    "conversationName",
                    "conversationID",
                    "turnNumber",
                    "utterance",
                ],
            )
            data_df_map[split_name] = df
        return data_df_map


def objectify(dset: ICCDataset) -> ICCDatasetSplits:
    assert isinstance(dset, ICCDataset)
    info = dset.info()
    # Pass the data into the split class and return that.
    obj = ICCDatasetSplits(info.data)
    return obj
