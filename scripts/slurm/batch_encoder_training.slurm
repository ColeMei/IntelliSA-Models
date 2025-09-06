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

# Load and verify environment
echo "🔧 Loading HPC environment..."
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh || exit 1
python --version && nvidia-smi && echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')" || exit 1

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
[ -f "${DATA_CACHE}/train.jsonl" ] && [ -f "${DATA_CACHE}/val.jsonl" ] || { echo "❌ Data files missing"; ls -la ${DATA_CACHE}/; exit 1; }
echo "✅ Data staged: $(wc -l < ${DATA_CACHE}/train.jsonl) train, $(wc -l < ${DATA_CACHE}/val.jsonl) val samples"

# Execute training
echo "🚀 Starting encoder training..."
cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Verify files exist
[ -f "scripts/train_models.py" ] && [ -f "${CONFIG_FILE}" ] || exit 1

srun --unbuffered python scripts/train_models.py \
    --approach encoder \
    --config ${CONFIG_FILE} \
    --train-path ${DATA_CACHE}/train.jsonl \
    --val-path ${DATA_CACHE}/val.jsonl \
    --output-dir ${MODEL_CHECKPOINTS} 2>&1 | tee ${TEMP_LOGS}/training_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log

[ $? -eq 0 ] || { echo "❌ Training failed - check: ${TEMP_LOGS}/training_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log"; cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true; exit 1; }

echo "✅ Training completed successfully!"

# Store results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERMANENT_MODEL_DIR="${PROJECT_ROOT}/models/experiments/encoder/${EXPERIMENT_NAME}_${TIMESTAMP}_job${SLURM_JOB_ID}"
RESULT_DIR="${PROJECT_ROOT}/results/experiments/encoder/${EXPERIMENT_NAME}_${TIMESTAMP}_job${SLURM_JOB_ID}"

mkdir -p ${PERMANENT_MODEL_DIR} ${RESULT_DIR} ${PROJECT_ROOT}/logs/training/
cp -r ${MODEL_CHECKPOINTS}/* ${PERMANENT_MODEL_DIR}/
cp ${TEMP_LOGS}/* ${RESULT_DIR}/ 2>/dev/null || true
cp ${CONFIG_FILE} ${RESULT_DIR}/config_used.yaml
cd ${RESULT_DIR} && ln -s ${PERMANENT_MODEL_DIR} model
cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true
cd ${PROJECT_ROOT}/models/experiments/encoder/ && rm -f ${EXPERIMENT_NAME}_latest 2>/dev/null || true && ln -s $(basename ${PERMANENT_MODEL_DIR}) ${EXPERIMENT_NAME}_latest

echo "✅ Results saved to: ${PERMANENT_MODEL_DIR}"
echo "📊 Storage: $(du -sh ${TRAINING_DIR} | cut -f1) used, $(du -sh ${PERMANENT_MODEL_DIR} | cut -f1) model"
echo "========================================="
echo "✅ Encoder training completed in $((SECONDS / 60)) minutes"
echo "Model: ${PROJECT_ROOT}/models/experiments/encoder/${EXPERIMENT_NAME}_latest"
echo "========================================="

# Note: No manual cleanup needed - /tmp is automatically cleaned when job ends
echo "🧹 Fast local storage will be automatically cleaned when job completes"
