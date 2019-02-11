#!/bin/bash
#SBATCH --account=def-siamakx
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-08:00
#SBATCH --output=%N-%j.out
module load miniconda3
source activate pytorch
python main.py -l -t -p -g 1 -n 50000 -k 10 -e 10 -b 20 -d 0 1 -lr 0.1
