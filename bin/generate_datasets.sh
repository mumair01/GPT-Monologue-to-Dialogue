#!/bin/bash
# @Author: Muhammad Umair
# @Date:   2022-08-22 12:36:11
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-16 13:58:24
#  Bash script to generate all processed data from raw data using the data
# library
# NOTE: Conda env must be active before running.
# PROJECT_PATH must be set. 

ICC_SCRIPT_PATH=${PROJECT_PATH}data_lib/icc.py
SPEAKER_IDENTITY_SCRIPT_PATH=${PROJECT_PATH}data_lib/speaker_identity_stims.py


declare -a variants=(no_labels special_labels)

for variant in "${variants[@]}"
do
    # Generate ICC Finetune data
    python ${ICC_SCRIPT_PATH} --path ${PROJECT_PATH}data/raw/ICC/finetune_experiments_julia/5_train_37_test_set/train \
        --outdir ${PROJECT_PATH}data/processed/ICC/5_train_37_test \
        --variant ${variant} --outfile train

    python ${ICC_SCRIPT_PATH} --path ${PROJECT_PATH}data/raw/ICC/finetune_experiments_julia/5_train_37_test_set/test \
        --outdir ${PROJECT_PATH}data/processed/ICC/5_train_37_test \
        --variant  ${variant} --outfile test

    python ${ICC_SCRIPT_PATH} --path ${PROJECT_PATH}data/raw/ICC/finetune_experiments_julia/28_train_14_test_set/train \
        --outdir ${PROJECT_PATH}data/processed/ICC/28_train_14_test \
        --variant  ${variant} --outfile train

    python ${ICC_SCRIPT_PATH} --path ${PROJECT_PATH}data/raw/ICC/finetune_experiments_julia/28_train_14_test_set/test \
        --outdir ${PROJECT_PATH}data/processed/ICC/28_train_14_test \
        --variant  ${variant} --outfile test

    # Generate the speaker identity stims inference data.
    python ${SPEAKER_IDENTITY_SCRIPT_PATH} --path ${PROJECT_PATH}data/raw/speaker_identity_stims \
        --outdir ${PROJECT_PATH}data/processed/speaker_identity_stims \
        --variant ${variant}

done

