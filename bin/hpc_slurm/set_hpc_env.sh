#!/bin/bash 

# Sets the paths required by the Tufts University HPC specifically. 
# NOTE: This should not be run, and will not work for, for non-Tufts clusters. 

# ------ These can be changed to construct the exported paths

## Root user path 
# NOTE: This should not be loaded using dirnmae due to HPC constraints


## Source the local env script first and then override for the HPC env. 
LOCAL_SET_ENV_PATH=../set_env.sh
source $LOCAL_SET_ENV_PATH

## Conda env.
CONDA_ENV_DIR_REL_PATH=condaenv 
CONDA_ENV_NAME=gpt_prod 

## Project paths 
PROJECT_DIR_REL_PATH=projects
PROJECT_DIR_NAME=gpt_test_5_16_23

# ## Scripts paths 
# SCRIPT_DIR_REL_PATH=scripts
# FINETUNE_SCRIPT_NAME=finetune.py
# INFERENCE_SCRIPT_NAME=inference.py

## Constructed paths
export USER_PATH=cluster/tufts/deruiterlab/mumair01 
export PROJECT_PATH=${USER_PATH}/${PROJECT_DIR_REL_PATH}/${PROJECT_DIR_NAME}
export PYTHON_ENV_PATH=${USER_PATH}/${CONDA_ENV_DIR_REL_PATH}/${CONDA_ENV_NAME}
export FINETUNE_SCRIPT_PATH=${PROJECT_PATH}/${SCRIPT_DIR_REL_PATH}/${FINETUNE_SCRIPT_NAME}
export INFERENCE_SCRIPT_PATH=${PROJECT_PATH}/${SCRIPT_DIR_REL_PATH}/${INFERENCE_SCRIPT_NAME}
## 
echo Paths set for Tufts HPC env:
echo USER_PATH=${USER_PATH}
echo PROJECT_PATH=${PROJECT_PATH}
echo PYTHON_ENV_PATH=${PYTHON_ENV_PATH}
echo FINETUNE_SCRIPT_PATH=${FINETUNE_SCRIPT_PATH}
echo INFERENCE_SCRIPT_PATH=${INFERENCE_SCRIPT_PATH}

## HPC Modules 
export ANACONDA_MOD=anaconda/2021.11
export CUDA_MODS="cuda/10.2 cudnn/7.1"



## Hydra args 
# export HYDRA_OVERWRITES="hydra.verbose=True"

# ## Dataset paths for hydra args
# export FICC537SL="finetune/icc_5_train_37_test_special_labels"
# export FICC2814SL="finetune/icc_28_train_14_test_special_labels"

# export FICC537NL="finetune/icc_5_train_37_test_no_labels"
# export FICC2814NL="finetune/icc_28_train_14_test_no_labels"

# export ISISSL="inference/speaker_identity_stims_special_labels"
# export ISISNL="inference/speaker_identity_stims_no_labels"

# ## Experiment names for hydra args
# export EXP_INF_TURNGPT="inference_turngpt"
# export EXP_INF_GPT2="inference_gpt2"

# export F_TURNGPT="finetune_turngpt"
# export F_GPT2="finetune_gpt2"

# module load cuda/11.0 cudnn/8.0.4-11.0

# Model Checkpoints for inference
# NOTE: These are loaded in the experiment.yaml files from the env. 
# since they are likely to change at a high frequency. 

# # model_checkpoint :/cluster/tufts/deruiterlab/mumair01/projects/gpt_monologue_dialogue/outputs/finetune_monologue_gpt/gpt2_icc_28_train_14_test_special_labels/2022-12-20_01-33-01/checkpoint-1160 
# export INF_GPT2_CKPT=gpt2
# # /cluster/tufts/deruiterlab/mumair01/projects/gpt_monologue_dialogue/outputs/finetune_turngpt/TurnGPT_LMHead_icc_28_train_14_test_no_labels/2022-12-20_02-33-02/finetune_turngpt_LMHead_icc_28_train_14_test_no_labels_32whwfm7/32whwfm7/checkpoints/epoch=6-step=987.ckpt
# export INF_TURNGPT_CKPT=gpt2

# ## HYDRA VARS.
# # Relative path to the hydra dir. from inside the scripts folder. 
# export HYDRA_CONFIG_RELATIVE_DIR=../conf/hydra
# # Name of the hydra default file
# export HYDRA_CONFIG_NAME=config
# ## WANDB VARS. 
# # Project name on wandb
# export WANDB_PROJECT=GPT-Monologue-Dialogue-Finetune
# # Project entity on wandb 
# export WANDB_ENTITY=gpt-monologue-dialogue
# # This initial mode for wandb - can be online, offline, or disabled
# # Link: https://able.bio/rhett/how-to-set-and-get-environment-variables-in-python--274rgt5
# export WANDB_INIT_MODE=offline