#### Sets all the 

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
# TODO: Add the path to the finetuned models here, or use gpt2 versions from 
# https://huggingface.co/docs/transformers/model_doc/gpt2#:~:text=Write%20With%20Transformer%20is%20a,small%20checkpoint%3A%20distilgpt%2D2.
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
