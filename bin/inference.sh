#!/usr/local/bin/bash 
# @Author: Muhammad Umair
# @Date:   2024-05-30 16:22:03
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2024-05-31 13:20:20


# for var_name in "${!GPT2_CKPTS[@]}" ; do
#   echo "${GPT2_CKPTS[$var_name]}"
# done



##########
# HELPER METHODS
##########

join_with_comma() {
  # Loop through arguments and print with comma
  for arg do
    echo -n "$arg,"  # Print argument without newline and followed by comma
  done
  # Remove trailing comma from the last element
  echo -n "${arg%,}"
}


PROJECT_PATH="$(dirname $(realpath $(dirname -- $0) ))"
TASK_SCRIPT_PATH=${PROJECT_PATH}/scripts/tasks_app.py

##########
# PATHS OF THE VARIOUS GPT2 MODELS 
##########

declare -A GPT2_CKPTS
GPT2_CKPTS["GPT2Labeled5Train37Test_noToks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled5Train37Test_noToksBwTurns_groupedConvs/30_05_2024__18_41_11/finetune"
GPT2_CKPTS["GPT2Labeled5Train37Test_noToks_noGrouping"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled5Train37Test_noToksBwTurns_noGrouping/30_05_2024__19_25_16/finetune"
GPT2_CKPTS["GPT2Labeled5Train37Test_toks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled5Train37Test_toksBwTurns_groupedConvs/30_05_2024__18_14_52/finetune"
GPT2_CKPTS["GPT2Labeled5Train37Test_toks_noGrouping"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled5Train37Test_toksBwTurns_noGrouping/30_05_2024__18_58_48/finetune"
GPT2_CKPTS["GPT2Labeled28Train14Test_noToks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled28Train14Test_noToksBwTurns_groupedConvs/30_05_2024__20_40_38/finetune"
GPT2_CKPTS["GPT2Labeled28Train14Test_noToks_noGrouping"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled28Train14Test_NoToksBwTurns_NoGrouping/30_05_2024__22_15_12/finetune"
GPT2_CKPTS["GPT2Labeled28Train14Test_toks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled28Train14Test_toksBwTurns_groupedConvs/30_05_2024__19_42_51/finetune"
GPT2_CKPTS["GPT2Labeled28Train14Test_toks_noGrouping"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_gpt2Labeled28Train14Test_toksBwTurns_noGrouping/30_05_2024__21_17_52/finetune"
# Combine all the models into a single comma separated string for hydra multi-run. 
GPT2_MODEL_CKPTS=$(join_with_comma "${GPT2_CKPTS[@]}")


##########
# PATHS OF THE VARIOUS TURNGPT MODELS 
##########

declare -A TURNGPT_CKPTS
TURNGPT_CKPTS["TurngptUnlabeled5Train37Test_toks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_turngptUnlabaled5Train37Test_toksBwTurns_groupedConvs/30_05_2024__22_52_01/finetune/lightning_logs/z7fcvm09/checkpoints/last.ckpt"
TURNGPT_CKPTS["TurngptUnlabeled5Train37Test_noToks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_turngptUnlabeled5Train37Test_noToksBwTurns_groupedConvs/30_05_2024__23_17_23/finetune/lightning_logs/ehh9lld8/checkpoints/last.ckpt"
TURNGPT_CKPTS["TurngptUnlabeled28Train14Test_noToks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_turngptUnlabeled28Train14Test_NoTokBwTurns_groupedConvs/31_05_2024__00_26_44/finetune/lightning_logs/0mcarlkg/checkpoints/last.ckpt"
TURNGPT_CKPTS["TurngptUnlabeled28Train14Test_toks_groupedConvs"]="/Users/muhammadumair/Documents/Repositories/mumair01-repos/GPT-Monologue-to-Dialogue/cache/pipelines/FT_turngptUnlabeled28Train14Test_ToksBwTurns_groupedConvs/30_05_2024__23_41_44/finetune/lightning_logs/fa57oii3/checkpoints/last.ckpt"

# NOTE: Specify all the turn model checkpoints here, separated by a comma. 
TURNGPT_MODEL_CKPTS=$(join_with_comma "${TURNGPT_CKPTS[@]}")


# Now, run all the combination of experiments

# Additional hydra parameters to overwrite. 
# NOTE: Even through each model is doing a sweep over add_eos_tok_bw_words and probability_methods, 
# they are not equally suited for each prediction!!! 
HYDRA_OVERWRITES="hydra.verbose=False dataset.add_eos_tok_bw_turns=True,False inference.pipeline.probability_method=words,eos"

HYDRA_ARGS_SWEEP=(
    "-m +experiment=inference/SILabeled ++model.load.model_checkpoint=${GPT2_MODEL_CKPTS} ${HYDRA_OVERWRITES}"
    "-m +experiment=inference/SIUnlabeled ++model.load.pretrained_model_name_or_path=${TURNGPT_MODEL_CKPTS} ${HYDRA_OVERWRITES}"

)



# Loop over each element in the list
for element in "${HYDRA_ARGS_SWEEP[@]}"; do
  # Your commands to process each element here
  python $TASK_SCRIPT_PATH $element ${HYDRA_OVERWRITES}
done