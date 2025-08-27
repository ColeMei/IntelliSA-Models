#!/bin/bash
#SBATCH --job-name=encoder_chef_training
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_encoder.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_encoder.err

# Load environment
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh

# Clean and prepare temp storage
echo "🧹 Preparing temp storage..."
rm -rf ${TEMP_ROOT}/active_training/data_cache/*
mkdir -p ${TEMP_ROOT}/active_training/data_cache

# Copy data to temp storage for better I/O
echo "📁 Staging data to temp storage..."
cp -r ${PROJECT_ROOT}/data/processed/* ${TEMP_ROOT}/active_training/data_cache/

# Run encoder training
echo "🚀 Starting encoder training..."
cd ${PROJECT_ROOT}

python scripts/train_models.py \
    --approach encoder \
    --model-name microsoft/codebert-base \
    --train-path ${TEMP_ROOT}/active_training/data_cache/chef_train.jsonl \
    --val-path ${TEMP_ROOT}/active_training/data_cache/chef_val.jsonl \
    --output-dir ${TEMP_ROOT}/active_training/model_checkpoints/encoder \
    --batch-size 8 \
    --num-epochs 3 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --save-steps 100 \
    --eval-steps 50

# Copy results back to permanent storage
echo "💾 Copying results to permanent storage..."
cp -r ${TEMP_ROOT}/active_training/model_checkpoints/encoder ${PROJECT_ROOT}/models/
cp ${TEMP_ROOT}/temp_logs/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true

# Clean up temp files to save space
echo "🧹 Cleaning temp files..."
rm -rf ${TEMP_ROOT}/active_training/data_cache/*

echo "✅ Encoder training completed!"
