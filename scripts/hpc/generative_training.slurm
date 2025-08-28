#!/bin/bash
#SBATCH --job-name=generative_chef_training
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-8:00:00
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_generative.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_generative.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qmmei@student.unimelb.edu.au

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start Time: $(date)"
echo "========================================="

# Load environment
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh

# Verify environment
echo "🔍 Verifying environment..."
python --version
nvidia-smi
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Clean and prepare temp storage
echo "🧹 Preparing temp storage..."
rm -rf ${TEMP_ROOT}/active_training/data_cache/*
mkdir -p ${TEMP_ROOT}/active_training/data_cache
mkdir -p ${TEMP_ROOT}/active_training/model_checkpoints/generative
mkdir -p ${TEMP_ROOT}/temp_logs

# Copy data to temp storage for better I/O
echo "📋 Staging data to temp storage..."
cp -r ${PROJECT_ROOT}/data/processed/* ${TEMP_ROOT}/active_training/data_cache/

# Verify data exists
if [[ ! -f "${TEMP_ROOT}/active_training/data_cache/chef_train.jsonl" ]]; then
    echo "❌ Training data not found! Exiting..."
    exit 1
fi

# Run generative training with error handling
echo "🚀 Starting generative training..."
cd ${PROJECT_ROOT}

# Set Python path explicitly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

srun --unbuffered python scripts/train_models.py \
    --approach generative \
    --config ${PROJECT_ROOT}/configs/generative_config.yaml \
    --train-path ${TEMP_ROOT}/active_training/data_cache/chef_train.jsonl \
    --val-path ${TEMP_ROOT}/active_training/data_cache/chef_val.jsonl \
    --output-dir ${TEMP_ROOT}/active_training/model_checkpoints/generative 2>&1 | tee ${TEMP_ROOT}/temp_logs/training_${SLURM_JOB_ID}.log

# Check if training completed successfully
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    # Still copy logs for debugging
    cp ${TEMP_ROOT}/temp_logs/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true
    exit $TRAIN_EXIT_CODE
fi

# Copy results back to permanent storage
echo "💾 Copying results to permanent storage..."
mkdir -p ${PROJECT_ROOT}/models/
mkdir -p ${PROJECT_ROOT}/logs/training/

cp -r ${TEMP_ROOT}/active_training/model_checkpoints/generative ${PROJECT_ROOT}/models/
cp ${TEMP_ROOT}/temp_logs/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true

# Clean up temp files to save space
echo "🧹 Cleaning temp files..."
rm -rf ${TEMP_ROOT}/active_training/data_cache/*
rm -rf ${TEMP_ROOT}/active_training/model_checkpoints/*

echo "========================================="
echo "✅ Generative training completed successfully!"
echo "End Time: $(date)"
echo "Job Duration: $((SECONDS / 60)) minutes"
echo "========================================="

# Package curated results snapshot for this job
./scripts/hpc/consolidate_results.sh "$SLURM_JOB_ID" --tar || echo "Consolidation skipped/failed; continuing."
