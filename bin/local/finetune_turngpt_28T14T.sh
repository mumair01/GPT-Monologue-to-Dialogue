# NOTE: set_env.sh must be sourced before running this script. 

# Requires the finetuning dataset and env to be specified.
DATASET=${FICC2814NL}
EXPERIMENT=${F_TURNGPT}
HYDRA_OVERWRITES="hydra.verbose=False"
HYDRA_ARGS="+experiment=${EXPERIMENT} +dataset=${DATASET} ${HYDRA_OVERWRITES}"

# Add the project directory to the pythonpath before running script
export PYTHONPATH=$PROJECT_PATH

python $FINETUNE_SCRIPT_PATH ${HYDRA_ARGS}