#!/bin/bash
#SBATCH -J 7_14_22_11_25_gpt_surprisal_inference_gpt2_large_ICC_experiment_28_train_14_test_07-12-2022_16-55-36_customDataset_speaker_identity_stims
#SBATCH --time=07-00:00:00 # maximum duration is 7 days
#SBATCH -p preempt #in 'preempt'
#SBATCH -N 1  #1 nodes
#SBATCH -n 32   #30 number of cores (number of threads)
#SBATCH --gres=gpu:1 # one P100 GPU, can request up to 6 on one node, total of 10, a100
#SBATCH --exclude=c1cmp[025-026]
#SBATCH -c 1 #1 cpu per task - leave this!
#SBATCH --mem=120g #requesting 60GB of RAM total
#SBATCH --output=./inference_reports/%x.%j.%N.out #saving standard output to file
#SBATCH --error=./inference_reports/%x.%j.%N.err # saving standard error to file
#SBATCH --mail-type=ALL # email optitions
#SBATCH --mail-user=muhammad.umair@tufts.edu

# Define paths
USER_PATH=/cluster/tufts/deruiterlab/mumair01/
SLURM_ENV_PATH=${USER_PATH}condaenv/gpt_proj
PROJECT_PATH=${USER_PATH}projects/gpt_monologue_dialogue/
SCRIPT_PATH=${PROJECT_PATH}src/inference/gpt_conditional_probs.py
CONFIG_PATH=${PROJECT_PATH}configs/inference/2.0-ConditionalProb-Inference-HPC.yaml

#load anaconda module
module load anaconda/2021.11

# NOTE: If not using a100 GPU, load the appropriate cuda version.
module load cuda/10.2 cudnn/7.1

# module load cuda/11.0 cudnn/8.0.4-11.0

# Get the GPU details
nvidia-smi

#activate conda environment
source activate $SLURM_ENV_PATH

python $SCRIPT_PATH --config $CONFIG_PATH

conda deactivate
