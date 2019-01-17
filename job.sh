#!/bin/bash
#SBATCH --account=def-siamakx  
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-05:00
#SBATCH --output=%N-%j.out
module load miniconda3
source activate pytorch
python main.py
