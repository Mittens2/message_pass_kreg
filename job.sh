#!/bin/bash
#SBATCH --account=def-siamakx  
#!/bin/bash
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=6  
#SBATCH --mem=32000
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out
module load miniconda3
source activate pytorch
python main.py
