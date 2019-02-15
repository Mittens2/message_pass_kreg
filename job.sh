#!/bin/bash
#SBATCH --account=def-siamakx
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-08:00
#SBATCH --output=%N-%j.out
module load miniconda3
source activate pytorch
python main.py -l -t -p -n 20000 -e 2 -b 20 -d 0 1 -lr 0.01 -da 0.2
