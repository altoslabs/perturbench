#!/bin/bash
#SBATCH --output=/cluster/scratch/fluebeck/perturbench_logs/log_files/hpo_%j.out
#SBATCH --error=/cluster/scratch/fluebeck/perturbench_logs/log_files/hpo_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=latent_embedding_hpo_frangieh21_no_gene_embedding

export HYDRA_FULL_ERROR=1
module load stack/2024-06 cuda/12.8.0
eval "$(micromamba shell hook -s bash)"
micromamba activate causalcell

# Submit HPO job
python src/perturbench/modelcore/train.py hpo=latent_embedding_hpo experiment=neurips2024/frangieh21/latent_embedding_best_params_frangieh21