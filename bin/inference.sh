#!/bin/bash
# @Author: Muhammad Umair
# @Date:   2024-05-30 16:22:03
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 17:42:51



PROJECT_PATH="$(dirname $(realpath $(dirname -- $0) ))"
TASK_SCRIPT_PATH=${PROJECT_PATH}/scripts/tasks_app.py

# NOTE: Specify all the gpt2 model checkpoints here, separated by a comma. 
GPT2_MODEL_CKPTS="gpt2"
TURNGPT_MODEL_CKPTS="gpt2"



# Now, run all the combination of experiments

# Additional hydra parameters to overwrite. 
HYDRA_OVERWRITES="hydra.verbose=False"

HYDRA_ARGS_SWEEP=(
    "-m +experiment=inference/SILabeled ++model.model_checkpoint=${GPT2_MODEL_CKPTS}"
    "-m +experiment=inference/SUnlabeled ++model.model_checkpoint=${GPT2_MODEL_CKPTS}"

)


# Loop over each element in the list
for element in "${HYDRA_ARGS_SWEEP[@]}"; do
  # Your commands to process each element here
  python $TASK_SCRIPT_PATH $element ${HYDRA_OVERWRITES}
done