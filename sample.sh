#!/bin/bash
#SBATCH --account=def-siamakx
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:00
#SBATCH --output=%N-%j.out
module load miniconda3
source activate pytorch
python main.py -l -p -n 50000
