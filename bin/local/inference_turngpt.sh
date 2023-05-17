# NOTE: set_env.sh must be sourced before running this script. 

# Requires the finetuning dataset and env to be specified.
DATASET=${ISISSL}
EXPERIMENT=${EXP_INF_TURNGPT}
HYDRA_OVERWRITES="hydra.verbose=False"
HYDRA_ARGS="+experiment=${EXPERIMENT} +dataset=${DATASET} ${HYDRA_OVERWRITES}"

# Add the project directory to the pythonpath before running script
export PYTHONPATH=$PROJECT_PATH

python $INFERENCE_SCRIPT_PATH ${HYDRA_ARGS}