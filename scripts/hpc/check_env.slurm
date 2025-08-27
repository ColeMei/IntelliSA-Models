#!/bin/bash
#SBATCH --job-name=env_check
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_envcheck.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_envcheck.err

set -euo pipefail

source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh

echo "=== Node & GPU (SLURM) ==="
echo "HOSTNAME: $(hostname)"
echo "NODELIST: ${SLURM_NODELIST}"
echo "GPUS: ${SLURM_STEP_GPUS:-}"

echo
echo "=== Running Python environment check ==="
srun -u python /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/scripts/check_env.py

echo
echo "=== Done ==="