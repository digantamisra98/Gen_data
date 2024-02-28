#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --job-name=gen_sdxl
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --reservation=ubuntu1804
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/sdxl-%j.out
#SBATCH --error=$SCRATCH/sdxl-%j.err

pyfile=/home/mila/d/diganta.misra/projects/Gen_data/generate.py

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/sparse

ulimit -Sn $(ulimit -Hn)

python3 $pyfile $1 $2
