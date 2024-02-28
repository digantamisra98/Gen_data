#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --job-name=LP_ViT_C10
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --reservation=ubuntu1804
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/lpvit-%j.out
#SBATCH --error=$SCRATCH/lpvit-%j.err

pyfile=/home/mila/d/diganta.misra/projects/generate.py

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/sparse

wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

ulimit -Sn $(ulimit -Hn)

wandb online

WANDB_CACHE_DIR=$SCRATCH

python3 $pyfile $1 $2
