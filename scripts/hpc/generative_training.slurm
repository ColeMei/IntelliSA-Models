#!/bin/bash
#SBATCH --job-name=generative_chef_training
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_generative.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_generative.err

# Load environment
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh

# Clean and prepare temp storage
echo "🧹 Preparing temp storage..."
rm -rf ${TEMP_ROOT}/active_training/data_cache/*
mkdir -p ${TEMP_ROOT}/active_training/data_cache

# Copy data to temp storage for better I/O
echo "📁 Staging data to temp storage..."
cp -r ${PROJECT_ROOT}/data/processed/* ${TEMP_ROOT}/active_training/data_cache/

# Run generative training (start with 7B)
echo "🚀 Starting generative training (7B model)..."
cd ${PROJECT_ROOT}

python scripts/train_models.py \
    --approach generative \
    --model-name codellama/CodeLlama-7b-hf \
    --train-path ${TEMP_ROOT}/active_training/data_cache/chef_train.jsonl \
    --val-path ${TEMP_ROOT}/active_training/data_cache/chef_val.jsonl \
    --output-dir ${TEMP_ROOT}/active_training/model_checkpoints/generative_7b \
    --batch-size 2 \
    --num-epochs 3 \
    --learning-rate 5e-5 \
    --warmup-steps 100 \
    --save-steps 100 \
    --eval-steps 50

# Copy results back to permanent storage
echo "💾 Copying results to permanent storage..."
cp -r ${TEMP_ROOT}/active_training/model_checkpoints/generative_7b ${PROJECT_ROOT}/models/generative/
cp ${TEMP_ROOT}/temp_logs/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true

# Clean up temp files to save space
echo "🧹 Cleaning temp files..."
rm -rf ${TEMP_ROOT}/active_training/data_cache/*

echo "✅ Generative training completed!"
