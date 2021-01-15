#!/bin/bash
#SBATCH --account=def-siamakx
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-08:00
# SBATCH --output=%N-%j.out
#SBATCH --array=4-9
module load miniconda3
source activate pytorch
python main.py -l -p -g 1 -n 100000 -k $SLURM_ARRAY_TASK_ID -e 20 -d 0 1 -lr 0.01 -th 100.0
