# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-15 08:58:15
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-15 09:01:31


class LanguageModel:
    """Abstract class the describes the methods required by a model """

    @property
    def tokenizer():
        raise NotImplementedError()

    def load(self, *args, **kwargs):
        raise NotImplementedError()

    def finetune(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
