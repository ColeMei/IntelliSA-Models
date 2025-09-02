#!/bin/bash
# SLURM parameters are set by the Python script that submits this job
# No need to specify them here as they will be overridden

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

# Get config file from command line argument
CONFIG_FILE="$1"
if [[ -z "$CONFIG_FILE" ]]; then
    echo "❌ No config file provided! Usage: sbatch batch_encoder_training.slurm <config_file>"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "📋 Using config file: $CONFIG_FILE"

# Extract experiment name from config file
EXPERIMENT_NAME=$(python -c "
import yaml, sys
try:
    with open('$CONFIG_FILE') as f:
        config = yaml.safe_load(f)
        metadata = config.get('experiment_metadata', {})
        print(metadata.get('name', '$(basename $CONFIG_FILE .yaml)'))
except Exception as e:
    print('$(basename $CONFIG_FILE .yaml)')
")

echo "🧪 Experiment: $EXPERIMENT_NAME"

# Set up fast local storage structure
echo "🏗️ Setting up fast local storage structure..."
FAST_STORAGE="/tmp"
TRAINING_DIR="${FAST_STORAGE}/batch_training_${SLURM_JOB_ID}"
DATA_CACHE="${TRAINING_DIR}/data_cache"
MODEL_CHECKPOINTS="${TRAINING_DIR}/model_checkpoints/${EXPERIMENT_NAME}"
TEMP_LOGS="${TRAINING_DIR}/logs"

# Clean and prepare fast local storage
rm -rf ${TRAINING_DIR} 2>/dev/null || true
mkdir -p ${DATA_CACHE}
mkdir -p ${MODEL_CHECKPOINTS}
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
echo "🚀 Starting batch encoder training on fast storage..."
cd ${PROJECT_ROOT}

# Set Python path explicitly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

srun --unbuffered python scripts/train_models.py \
    --approach encoder \
    --config ${CONFIG_FILE} \
    --train-path ${DATA_CACHE}/chef_train.jsonl \
    --val-path ${DATA_CACHE}/chef_val.jsonl \
    --output-dir ${MODEL_CHECKPOINTS} 2>&1 | tee ${TEMP_LOGS}/training_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log

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

# Create timestamped directories for permanent storage
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERMANENT_MODEL_DIR="${PROJECT_ROOT}/models/experiments/encoder/${EXPERIMENT_NAME}_${TIMESTAMP}_job${SLURM_JOB_ID}"
RESULT_DIR="${PROJECT_ROOT}/results/experiments/encoder/${EXPERIMENT_NAME}_${TIMESTAMP}_job${SLURM_JOB_ID}"

echo "💾 Copying results to permanent storage..."
echo "  Model destination: ${PERMANENT_MODEL_DIR}"
echo "  Results destination: ${RESULT_DIR}"

# Create directories
mkdir -p ${PERMANENT_MODEL_DIR}
mkdir -p ${RESULT_DIR}
mkdir -p ${PROJECT_ROOT}/logs/training/

# Copy model to timestamped location in models directory
cp -r ${MODEL_CHECKPOINTS}/* ${PERMANENT_MODEL_DIR}/

# Create symlink in results directory instead of copying (saves storage)
ln -s ../../../models/experiments/encoder/$(basename ${PERMANENT_MODEL_DIR}) ${RESULT_DIR}/model

# Copy logs and config to results directory
cp ${TEMP_LOGS}/* ${RESULT_DIR}/ 2>/dev/null || true
cp ${CONFIG_FILE} ${RESULT_DIR}/config_used.yaml

# Copy logs to training logs directory
cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true

# Create symlink to latest model for this experiment
LATEST_DIR="${PROJECT_ROOT}/models/experiments/encoder/${EXPERIMENT_NAME}_latest"
rm -f ${LATEST_DIR} 2>/dev/null || true
ln -s $(basename ${PERMANENT_MODEL_DIR}) ${LATEST_DIR}

echo "✅ Results copied successfully!"
echo "  Latest model for ${EXPERIMENT_NAME}: ${LATEST_DIR} -> $(basename ${PERMANENT_MODEL_DIR})"

# Display storage usage summary
echo "📊 Storage usage summary:"
echo "  Fast storage used: $(du -sh ${TRAINING_DIR} | cut -f1)"
echo "  Final model size: $(du -sh ${PERMANENT_MODEL_DIR} | cut -f1)"
echo "  Results directory size: $(du -sh ${RESULT_DIR} | cut -f1)"
echo "  💡 Model files stored only in models/ directory (symlinked in results/)"

echo "========================================="
echo "✅ Batch encoder training completed successfully!"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "End Time: $(date)"
echo "Job Duration: $((SECONDS / 60)) minutes"
echo "Model saved to: ${PERMANENT_MODEL_DIR}"
echo "Results archived in: ${RESULT_DIR}"
echo "========================================="

# Note: No manual cleanup needed - /tmp is automatically cleaned when job ends
echo "🧹 Fast local storage will be automatically cleaned when job completes"
