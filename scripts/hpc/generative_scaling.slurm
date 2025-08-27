#!/bin/bash
#SBATCH --job-name=generative_scaling
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_scaling.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_scaling.err

# Load environment
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh

# Clean and prepare temp storage
rm -rf ${TEMP_ROOT}/active_training/data_cache/*
mkdir -p ${TEMP_ROOT}/active_training/data_cache

# Copy data to temp storage
cp -r ${PROJECT_ROOT}/data/processed/* ${TEMP_ROOT}/active_training/data_cache/

cd ${PROJECT_ROOT}

# Train 13B model
echo "🚀 Starting 13B model training..."
python scripts/train_models.py \
    --approach generative \
    --model-name codellama/CodeLlama-13b-hf \
    --train-path ${TEMP_ROOT}/active_training/data_cache/chef_train.jsonl \
    --val-path ${TEMP_ROOT}/active_training/data_cache/chef_val.jsonl \
    --output-dir ${TEMP_ROOT}/active_training/model_checkpoints/generative_13b \
    --batch-size 1 \
    --num-epochs 3

# Copy 13B results back
cp -r ${TEMP_ROOT}/active_training/model_checkpoints/generative_13b ${PROJECT_ROOT}/models/generative/

# Train 32B model (if resources allow)
echo "🚀 Starting 32B model training..."
python scripts/train_models.py \
    --approach generative \
    --model-name codellama/CodeLlama-32b-hf \
    --train-path ${TEMP_ROOT}/active_training/data_cache/chef_train.jsonl \
    --val-path ${TEMP_ROOT}/active_training/data_cache/chef_val.jsonl \
    --output-dir ${TEMP_ROOT}/active_training/model_checkpoints/generative_32b \
    --batch-size 1 \
    --num-epochs 2

# Copy 32B results back
cp -r ${TEMP_ROOT}/active_training/model_checkpoints/generative_32b ${PROJECT_ROOT}/models/generative/

# Clean up temp files
rm -rf ${TEMP_ROOT}/active_training/data_cache/*

echo "✅ Model scaling completed!"
