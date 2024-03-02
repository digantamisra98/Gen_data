#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --job-name=gen_floyd_domainnet
#SBATCH --mem=36G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/floyd-%j.out
#SBATCH --error=$SCRATCH/floyd-%j.err

pyfile=/home/mila/d/diganta.misra/projects/Gen_data/generate_floyd.py

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/floyd

ulimit -Sn $(ulimit -Hn)

python3 $pyfile $1 $2 $3
