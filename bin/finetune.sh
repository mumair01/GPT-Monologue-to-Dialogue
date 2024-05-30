#!/bin/bash
# @Author: Muhammad Umair
# @Date:   2024-05-30 16:22:00
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-30 17:42:32


PROJECT_PATH="$(dirname $(realpath $(dirname -- $0) ))"
TASK_SCRIPT_PATH=${PROJECT_PATH}/scripts/tasks_app.py

echo $TASK_SCRIPT_PATH

# Now, run all the combination of experiments

# Additional hydra parameters to overwrite. 
HYDRA_OVERWRITES="hydra.verbose=False"

# List of arguments to run each hydra experiment with. 
# NOTE: In total, this should generate 12 finetuned models!
HYDRA_ARGS_SWEEP=(
    # We will multi-run GPT-2 fine-tuning and train 4 models where the 
    # training data is either grouped by conversation or not and there 
    # is an explicit end of turn tok between all turns in the training data 
    # (as is the method specified by reviewer 2). 
    # Overall, this will create eight models (4 for each training dataset)

    "-m +experiment=finetune/gpt2ICCLabeled5Train37Test,finetune/gpt2ICCLabeled28Train14Test dm.group_by_conversation=True,False dataset.add_eos_tok_bw_turns=True,False ++tracker.mode=disabled ++model.finetune.num_train_epochs=30"
    
    # This is the multi-run experiment for TurnGPT finetuning. Here, we only 
    # create two models that alternate between whether or not the eot token 
    # is added between all turns. Overall, this will create 4 models (2 for 
    # each training dataset)
    "-m +experiment=finetune/turngptUnlabeled5Train37Test,finetune/turngptUnlabeled28Train14Test dataset.add_eos_tok_bw_turns=True,False ++tracker.mode=disabled ++model.finetune.max_epochs=30"
)

# Loop over each element in the list
for element in "${HYDRA_ARGS_SWEEP[@]}"; do
  # Your commands to process each element here
  python $TASK_SCRIPT_PATH $element ${HYDRA_OVERWRITES}
done


# HYDRA_ARGS="+experiment=${EXPERIMENT} +dataset=${DATASET} ${HYDRA_OVERWRITES}"
