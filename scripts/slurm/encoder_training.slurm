#!/bin/bash
#SBATCH --job-name=encoder_chef_training
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-4:00:00
#SBATCH --tmp=30GB
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_encoder.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_encoder.err
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
echo "Fast Storage: $(df -h /tmp | tail -1 | awk '{print $2" available, "$4" free"}')"
echo "Start Time: $(date)"
echo "========================================="

# Load environment
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh

# Verify environment
echo "🔍 Verifying environment..."
python --version
nvidia-smi
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set up fast local storage structure
echo "🏗️ Setting up fast local storage structure..."
FAST_STORAGE="/tmp"
TRAINING_DIR="${FAST_STORAGE}/training_${SLURM_JOB_ID}"
DATA_CACHE="${TRAINING_DIR}/data_cache"
MODEL_CHECKPOINTS="${TRAINING_DIR}/model_checkpoints"
TEMP_LOGS="${TRAINING_DIR}/logs"

# Clean and prepare fast local storage
rm -rf ${TRAINING_DIR} 2>/dev/null || true
mkdir -p ${DATA_CACHE}
mkdir -p ${MODEL_CHECKPOINTS}/encoder
mkdir -p ${TEMP_LOGS}

# Copy data to fast local storage for better I/O
echo "📋 Staging data to fast local storage..."
cp -r ${PROJECT_ROOT}/data/processed/* ${DATA_CACHE}/

# Verify data exists on fast storage
if [[ ! -f "${DATA_CACHE}/chef_train.jsonl" ]]; then
    echo "❌ Training data not found on fast storage! Exiting..."
    exit 1
fi

echo "✅ Data staged successfully to fast storage"
echo "  Training data: ${DATA_CACHE}/chef_train.jsonl"
echo "  Validation data: ${DATA_CACHE}/chef_val.jsonl"

# Run encoder training with error handling
echo "🚀 Starting encoder training on fast storage..."
cd ${PROJECT_ROOT}

# Set Python path explicitly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

srun --unbuffered python scripts/train_models.py \
    --approach encoder \
    --config ${PROJECT_ROOT}/configs/encoder_config.yaml \
    --train-path ${DATA_CACHE}/chef_train.jsonl \
    --val-path ${DATA_CACHE}/chef_val.jsonl \
    --output-dir ${MODEL_CHECKPOINTS}/encoder 2>&1 | tee ${TEMP_LOGS}/training_${SLURM_JOB_ID}.log

# Check if training completed successfully
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    # Still copy logs for debugging
    mkdir -p ${PROJECT_ROOT}/logs/training/
    cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true
    exit $TRAIN_EXIT_CODE
fi

echo "✅ Training completed successfully!"

# Create timestamped directories for permanent storage (no overwriting)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERMANENT_MODEL_DIR="${PROJECT_ROOT}/models/encoder_${TIMESTAMP}_job${SLURM_JOB_ID}"
RESULT_DIR="${PROJECT_ROOT}/results/run_${SLURM_JOB_ID}_${TIMESTAMP}"

echo "💾 Copying results to permanent storage..."
echo "  Model destination: ${PERMANENT_MODEL_DIR}"
echo "  Results destination: ${RESULT_DIR}"

# Create directories
mkdir -p ${PERMANENT_MODEL_DIR}
mkdir -p ${RESULT_DIR}
mkdir -p ${PROJECT_ROOT}/logs/training/

# Copy model to timestamped location (preserve all training runs)
cp -r ${MODEL_CHECKPOINTS}/encoder/* ${PERMANENT_MODEL_DIR}/

# Copy only logs and config to results directory (no model duplication)
cp ${TEMP_LOGS}/* ${RESULT_DIR}/ 2>/dev/null || true
cp ${PROJECT_ROOT}/configs/encoder_config.yaml ${RESULT_DIR}/config_used.yaml

# Create symlink from results to model (saves space)
cd ${RESULT_DIR}
ln -s ${PERMANENT_MODEL_DIR} model

# Copy logs to training logs directory
cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true

# Create symlink to latest model (for easy access)
cd ${PROJECT_ROOT}/models/
rm -f encoder_latest 2>/dev/null || true
ln -s $(basename ${PERMANENT_MODEL_DIR}) encoder_latest

echo "✅ Results copied successfully!"
echo "  Latest model: ${PROJECT_ROOT}/models/encoder_latest -> $(basename ${PERMANENT_MODEL_DIR})"
echo "  All models preserved in: ${PROJECT_ROOT}/models/"

# Display storage usage summary
echo "📊 Storage usage summary:"
echo "  Fast storage used: $(du -sh ${TRAINING_DIR} | cut -f1)"
echo "  Final model size: $(du -sh ${PERMANENT_MODEL_DIR} | cut -f1)"

echo "========================================="
echo "✅ Encoder training completed successfully!"
echo "End Time: $(date)"
echo "Job Duration: $((SECONDS / 60)) minutes"
echo "Model saved to: ${PERMANENT_MODEL_DIR}"
echo "Results archived in: ${RESULT_DIR}"
echo "========================================="

# Note: No manual cleanup needed - /tmp is automatically cleaned when job ends
echo "🧹 Fast local storage will be automatically cleaned when job completes"