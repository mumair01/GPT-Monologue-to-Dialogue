# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2024-05-19 15:32:07
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-22 07:44:00


from LLMDialogue.datasets.ICC import load_ICC
from LLMDialogue.datasets.speakerIdentity import load_speakerIdentity
from LLMDialogue.datasets.dsets import DatasetBuilder, DatasetConfig


_DSET_LOADERS = {"ICC": load_ICC, "speakerIdentity": load_speakerIdentity}


def load(
    dset_name: str,
    config_name: str,
    **loader_kwargs,
) -> DatasetBuilder:
    assert dset_name in list(
        _DSET_LOADERS.keys()
    ), f"Unsupported dset, pick one from: {list(_DSET_LOADERS.keys())}"

    dset = _DSET_LOADERS[dset_name](config_name=config_name, **loader_kwargs)
    return dset
