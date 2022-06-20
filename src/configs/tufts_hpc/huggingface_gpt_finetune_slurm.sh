#!/bin/bash
#SBATCH -J gpt_monologue_dialogue_30_epochs_6_20_22_job_1 #job name
#SBATCH --time=07-00:00:00 # maximum duration is 7 days
#SBATCH -p preempt #in 'preempt'
#SBATCH -N 1  #1 nodes
#SBATCH -n 16  #30 number of cores (number of threads)
#SBATCH --gres=gpu:1 # one P100 GPU, can request up to 6 on one node, total of 10, a100
#SBATCH --exclude=c1cmp[025-026]
#SBATCH -c 1 #1 cpu per task - leave this!
#SBATCH --mem=60g #requesting 60GB of RAM total
#SBATCH --output=myjob.%j.%N.out #saving standard output to file
#SBATCH --error=myjob.%j.%N.err # saving standard error to file
#SBATCH --mail-type=ALL # email optitions
#SBATCH --mail-user=muhammad.umair@tufts.edu

#load anaconda module
module load anaconda/3

#load anaconda module
module load cuda/10.0

#activate conda environment
source activate /cluster/tufts/deruiterlab/mumair01/condaenv/jpt

python /cluster/tufts/deruiterlab/mumair01/projects/jpt/scripts/finetune_pt_slurm.py

conda deactivate