
# NOTE: set_env.sh must be sourced before running this script. 

# Requires the finetuning dataset and env to be specified.
DATASET=${ISISSL}
EXPERIMENT=${EXP_INF_GPT2}
HYDRA_OVERWRITES="hydra.verbose=False"
HYDRA_ARGS="+experiment=${EXPERIMENT} +dataset=${DATASET} ${HYDRA_OVERWRITES}"

# Add the project directory to the pythonpath before running script
export PYTHONPATH=$PROJECT_PATH

echo $HYDRA_ARGS

python $INFERENCE_SCRIPT_PATH ${HYDRA_ARGS}