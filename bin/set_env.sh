#!/bin/bash
# @Author: Muhammad Umair
# @Date:   2023-05-16 13:40:14
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-16 21:06:48


## Scripts paths 
SCRIPT_DIR_REL_PATH=scripts
FINETUNE_SCRIPT_NAME=finetune.py
INFERENCE_SCRIPT_NAME=inference.py

# Paths
export PROJECT_PATH="$(dirname $(realpath $(dirname -- $0) ))"
export FINETUNE_SCRIPT_PATH=${PROJECT_PATH}/${SCRIPT_DIR_REL_PATH}/${FINETUNE_SCRIPT_NAME}
export INFERENCE_SCRIPT_PATH=${PROJECT_PATH}/${SCRIPT_DIR_REL_PATH}/${INFERENCE_SCRIPT_NAME}

echo "set PROJECT_PATH=$PROJECT_PATH"
echo FINETUNE_SCRIPT_PATH=${FINETUNE_SCRIPT_PATH}
echo INFERENCE_SCRIPT_PATH=${INFERENCE_SCRIPT_PATH}

## Hydra args 
export HYDRA_OVERWRITES="hydra.verbose=True"

## Dataset paths for hydra args
export FICC537SL="finetune/icc_5_train_37_test_special_labels"
export FICC2814SL="finetune/icc_28_train_14_test_special_labels"

export FICC537NL="finetune/icc_5_train_37_test_no_labels"
export FICC2814NL="finetune/icc_28_train_14_test_no_labels"

export ISISSL="inference/speaker_identity_stims_special_labels"
export ISISNL="inference/speaker_identity_stims_no_labels"

## Experiment names for hydra args
export EXP_INF_TURNGPT="inference_turngpt"
export EXP_INF_GPT2="inference_gpt2"

export F_TURNGPT="finetune_turngpt"
export F_GPT2="finetune_gpt2"

# Model Checkpoints for inference
# NOTE: These are loaded in the experiment.yaml files from the env. 
# since they are likely to change at a high frequency. 
export INF_GPT2_CKPT=gpt2
export INF_TURNGPT_CKPT=gpt2

## HYDRA VARS.
# Relative path to the hydra dir. from inside the scripts folder. 
export HYDRA_CONFIG_RELATIVE_DIR=../conf/hydra
# Name of the hydra default file
export HYDRA_CONFIG_NAME=config
## WANDB VARS. 
# Project name on wandb
export WANDB_PROJECT=GPT-Monologue-Dialogue-Finetune
# Project entity on wandb 
export WANDB_ENTITY=gpt-monologue-dialogue
# This initial mode for wandb - can be online, offline, or disabled
# Link: https://able.bio/rhett/how-to-set-and-get-environment-variables-in-python--274rgt5
export WANDB_INIT_MODE=disabled

# Provide all files in the local folder with run access
chmod +x ${PROJECT_PATH}/bin/local/*.sh