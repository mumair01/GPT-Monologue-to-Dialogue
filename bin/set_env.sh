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


# Source all configs 
source ./set_configs.sh

# Provide all files in the local folder with run access
chmod +x ${PROJECT_PATH}/bin/local/*.sh