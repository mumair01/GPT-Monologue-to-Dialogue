#!/bin/bash
#SBATCH -J 7_6_22_3_28_gpt_finetune_textDataset #job name
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
SLURM_ENV_PATH=${USER_PATH}condaenv/gpt_proj
PROJECT_PATH=${USER_PATH}projects/gpt_monologue_dialogue/
SCRIPT_PATH=${PROJECT_PATH}src/finetuning/1.0-GPT-finetune.py
CONFIG_PATH=${PROJECT_PATH}configs/finetuning/1.0-GPT-Finetune-TextDataset-HPC.yaml

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
