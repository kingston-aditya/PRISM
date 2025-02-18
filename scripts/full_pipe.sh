#!/bin/bash
#SBATCH -c 32                                
#SBATCH --gres=gpu:rtxa4000:8 
#SBATCH --time=2-23:00:00       
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --mail-user=asarkar6@umd.edu
#SBATCH --mail-type=ALL

# Setup your environment (e.g., module load, source activate)
source activate base
conda activate textsam

# Run your code (e.g., python train.py)
python /nfshomes/asarkar6/PRISM/main.py 6