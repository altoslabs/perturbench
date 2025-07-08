#!/bin/bash
#SBATCH --output=/cluster/scratch/fluebeck/perturbench_logs/log_files/info_%j.out
#SBATCH --error=/cluster/scratch/fluebeck/perturbench_logs/log_files/info_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g


export HYDRA_FULL_ERROR=1
module load stack/2024-06 cuda/12.8.0
eval "$(micromamba shell hook -s bash)"
micromamba activate causalcell
python te.py #src/perturbench/modelcore/train.py experiment=neurips2024/frangieh21/latent_embedding_best_params_frangieh21
