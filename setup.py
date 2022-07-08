# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-07-08 10:15:25
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-07-08 11:59:11


from setuptools import setup, find_packages

setup_info = dict(
    name='gpt:monologue_to_dialogue',
    version="0.0.1",
    author="Muhammad Umair",
    author_email="muhammad.umair@tufts.edu",
    project_urls={
        "Source" : "https://github.com/mumair01/GPT-Monologue-to-Dialogue",
        "Trainer" : "https://github.com/mumair01/GPT-Monologue-to-Dialogue/issues"
    },
    packages=['src'] + ['src.' + pkg for pkg in find_packages('src')],

)

setup(**setup_info)