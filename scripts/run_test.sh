#!/bin/bash
#SBATCH -c 32                                
#SBATCH --gres=gpu:rtxa6000:4      
#SBATCH --time=2-23:00:00       
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger

# Setup your environment (e.g., module load, source activate)
conda activate textsam

# Run your code (e.g., python train.py)
python /nfshomes/asarkar6/PRISM/tasks/data_gen.py