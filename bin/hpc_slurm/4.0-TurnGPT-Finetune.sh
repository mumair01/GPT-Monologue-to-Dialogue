#!/bin/bash
#SBATCH -J turn_gpt_finetune #job name
#SBATCH --time=07-00:00:00 # maximum duration is 7 days
#SBATCH -p preempt #in 'preempt'
#SBATCH -N 1  #1 nodes
#SBATCH -n 32   #30 number of cores (number of threads)
#SBATCH --gres=gpu:1 # one P100 GPU, can request up to 6 on one node, total of 10, a100
#SBATCH --exclude=c1cmp[025-026]
#SBATCH -c 1 #1 cpu per task - leave this!
#SBATCH --mem=120g #requesting 60GB of RAM total
#SBATCH --output=./finetuning_reports/%x.%j.%N.out #saving standard output to file
#SBATCH --error=./finetuning_reports/%x.%j.%N.err # saving standard error to file
#SBATCH --mail-type=ALL # email optitions
#SBATCH --mail-user=muhammad.umair@tufts.edu

# Define paths
USER_PATH=/cluster/tufts/deruiterlab/mumair01/
PYTHON_ENV_PATH=${USER_PATH}condaenv/gpt_proj
PROJECT_PATH=${USER_PATH}projects/gpt_monologue_dialogue/
SCRIPT_PATH=${PROJECT_PATH}gpt_dialogue/scripts/finetune_turngpt.py

# Requires the finetuning dataset and env to be specified.
HYDRA_ENV="hpc"
DATASET="finetune/icc_5_train_37_test"
HYDRA_OVERWRITES=""
HYDRA_ARGS="+env=${HYDRA_ENV} +dataset=${DATASET} ${HYDRA_OVERWRITES}"

#load anaconda module
module load anaconda/2021.11

# NOTE: If not using a100 GPU, load the appropriate cuda version.
module load cuda/10.2 cudnn/7.1

# module load cuda/11.0 cudnn/8.0.4-11.0

# Get the GPU details
nvidia-smi

#activate conda environment
source activate $PYTHON_ENV_PATH

# Add the project directory to the pythonpath before running script
export PYTHONPATH=$PROJECT_PATH


python $SCRIPT_PATH ${HYDRA_ARGS}

conda deactivate
