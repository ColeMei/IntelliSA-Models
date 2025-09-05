#!/bin/bash
#SBATCH --job-name=generative_training
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G          # Higher memory for large generative models
#SBATCH --time=0-8:00:00   # Longer time for generative training
#SBATCH --tmp=50GB         # Larger temp storage for model checkpoints
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_generative.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_generative.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qmmei@student.unimelb.edu.au

# Job info
echo "========================================="
echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID) | Node: $SLURM_JOB_NODELIST"
echo "Resources: $SLURM_CPUS_PER_TASK CPUs, GPU: $CUDA_VISIBLE_DEVICES, ${SLURM_MEM_PER_NODE}MB RAM"
echo "Start: $(date)"
echo "========================================="

# Load and verify environment
echo "🔧 Loading HPC environment..."
source /data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/environments/setup_hpc_env.sh || exit 1
python --version && nvidia-smi && echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')" || exit 1

# Setup fast local storage
echo "🏗️ Setting up fast local storage..."
FAST_STORAGE="/tmp"
TRAINING_DIR="${FAST_STORAGE}/training_${SLURM_JOB_ID}"
DATA_CACHE="${TRAINING_DIR}/data_cache"
MODEL_CHECKPOINTS="${TRAINING_DIR}/model_checkpoints"
TEMP_LOGS="${TRAINING_DIR}/logs"

rm -rf ${TRAINING_DIR} 2>/dev/null || true
mkdir -p ${DATA_CACHE} ${MODEL_CHECKPOINTS}/generative ${TEMP_LOGS}

# Stage and verify data
echo "📋 Staging data..."
cp -r ${PROJECT_ROOT}/data/processed/* ${DATA_CACHE}/
[ -f "${DATA_CACHE}/train.jsonl" ] && [ -f "${DATA_CACHE}/val.jsonl" ] || { echo "❌ Data files missing"; ls -la ${DATA_CACHE}/; exit 1; }
echo "✅ Data staged: $(wc -l < ${DATA_CACHE}/train.jsonl) train, $(wc -l < ${DATA_CACHE}/val.jsonl) val samples"

# Execute training
echo "🚀 Starting generative training..."
cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Verify files exist
[ -f "scripts/train_models.py" ] && [ -f "${PROJECT_ROOT}/configs/generative_config.yaml" ] || exit 1

srun --unbuffered python scripts/train_models.py \
    --approach generative \
    --config ${PROJECT_ROOT}/configs/generative_config.yaml \
    --train-path ${DATA_CACHE}/train.jsonl \
    --val-path ${DATA_CACHE}/val.jsonl \
    --output-dir ${MODEL_CHECKPOINTS}/generative 2>&1 | tee ${TEMP_LOGS}/training_${SLURM_JOB_ID}.log

[ $? -eq 0 ] || { echo "❌ Training failed - check: ${TEMP_LOGS}/training_${SLURM_JOB_ID}.log"; cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true; exit 1; }

echo "✅ Training completed successfully!"

# Store results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERMANENT_MODEL_DIR="${PROJECT_ROOT}/models/generative_${TIMESTAMP}_job${SLURM_JOB_ID}"
RESULT_DIR="${PROJECT_ROOT}/results/run_${SLURM_JOB_ID}_${TIMESTAMP}"

mkdir -p ${PERMANENT_MODEL_DIR} ${RESULT_DIR} ${PROJECT_ROOT}/logs/training/
cp -r ${MODEL_CHECKPOINTS}/generative/* ${PERMANENT_MODEL_DIR}/
cp ${TEMP_LOGS}/* ${RESULT_DIR}/ 2>/dev/null || true
cp ${PROJECT_ROOT}/configs/generative_config.yaml ${RESULT_DIR}/config_used.yaml
cd ${RESULT_DIR} && ln -s ${PERMANENT_MODEL_DIR} model
cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/training/ 2>/dev/null || true
cd ${PROJECT_ROOT}/models/ && rm -f generative_latest 2>/dev/null || true && ln -s $(basename ${PERMANENT_MODEL_DIR}) generative_latest

echo "✅ Results saved to: ${PERMANENT_MODEL_DIR}"
echo "📊 Storage: $(du -sh ${TRAINING_DIR} | cut -f1) used, $(du -sh ${PERMANENT_MODEL_DIR} | cut -f1) model"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk '{print $1"/"$2" MB"}')"
echo "========================================="
echo "✅ Generative training completed in $((SECONDS / 60)) minutes"
echo "Model: ${PROJECT_ROOT}/models/generative_latest"
echo "========================================="

# Note: No manual cleanup needed - /tmp is automatically cleaned when job ends
echo "🧹 Fast local storage will be automatically cleaned when job completes"