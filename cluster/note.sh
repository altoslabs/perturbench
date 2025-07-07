#!/bin/bash
#SBATCH --output=logs/log_files/info_%j.out
#SBATCH --error=logs/log_files/info_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g


module load stack/2024-06 cuda/12.8.0
eval "$(micromamba shell hook -s bash)"
micromamba activate causalcell
python src/perturbench/modelcore/train.py experiment=neurips2024/frangieh21/cpa_best_params_frangieh21
