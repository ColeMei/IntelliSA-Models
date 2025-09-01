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

# Get config file from command line argument
CONFIG_FILE="$1"
if [[ -z "$CONFIG_FILE" ]]; then
    echo "❌ No config file provided! Usage: sbatch batch_evaluate_models.slurm <config_file>"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "📋 Using config file: $CONFIG_FILE"

# Set up fast local storage structure
echo "🗂️ Setting up fast local storage structure..."
FAST_STORAGE="/tmp"
EVAL_DIR="${FAST_STORAGE}/batch_eval_${SLURM_JOB_ID}"
DATA_CACHE="${EVAL_DIR}/data_cache"
RESULTS_CACHE="${EVAL_DIR}/results"
TEMP_LOGS="${EVAL_DIR}/logs"

# Clean and prepare fast local storage
rm -rf ${EVAL_DIR} 2>/dev/null || true
mkdir -p ${DATA_CACHE}
mkdir -p ${RESULTS_CACHE}
mkdir -p ${TEMP_LOGS}

# Copy test data to fast local storage
echo "📋 Staging test data to fast local storage..."
cp -r ${PROJECT_ROOT}/data/processed/chef_test.jsonl ${DATA_CACHE}/

# Verify test data exists
if [[ ! -f "${DATA_CACHE}/chef_test.jsonl" ]]; then
    echo "❌ Test data not found on fast storage! Exiting..."
    exit 1
fi

echo "✅ Test data staged successfully"

# Set Python path explicitly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

cd ${PROJECT_ROOT}

# Run encoder evaluation only
echo "🚀 Evaluating encoder model..."
srun --unbuffered python scripts/evaluate_models.py \
    --approach encoder \
    --config ${CONFIG_FILE} \
    --test-path ${DATA_CACHE}/chef_test.jsonl \
    --output-dir ${RESULTS_CACHE}/encoder_eval 2>&1 | tee ${TEMP_LOGS}/encoder_evaluation_${SLURM_JOB_ID}.log

EVAL_EXIT_CODE=$?
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "❌ Encoder evaluation failed with exit code $EVAL_EXIT_CODE"
    # Still copy logs for debugging
    mkdir -p ${PROJECT_ROOT}/logs/evaluation/
    cp -r ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/evaluation/ 2>/dev/null || true
    exit $EVAL_EXIT_CODE
fi

echo "✅ Encoder evaluation completed successfully!"

# Create timestamped directories for permanent storage
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERMANENT_RESULTS_DIR="${PROJECT_ROOT}/results/evaluation_${TIMESTAMP}_job${SLURM_JOB_ID}"
EVALUATION_LOGS_DIR="${PROJECT_ROOT}/logs/evaluation/"

echo "💾 Copying evaluation results to permanent storage..."
echo "  Results destination: ${PERMANENT_RESULTS_DIR}"

# Create directories
mkdir -p ${PERMANENT_RESULTS_DIR}
mkdir -p ${EVALUATION_LOGS_DIR}

# Copy all results to permanent storage
cp -r ${RESULTS_CACHE}/* ${PERMANENT_RESULTS_DIR}/
cp ${TEMP_LOGS}/* ${PERMANENT_RESULTS_DIR}/ 2>/dev/null || true

# Copy logs to evaluation logs directory
cp ${TEMP_LOGS}/* ${EVALUATION_LOGS_DIR}/ 2>/dev/null || true

# Create symlink to latest evaluation results
cd ${PROJECT_ROOT}/results/
rm -f evaluation_latest 2>/dev/null || true
ln -s $(basename ${PERMANENT_RESULTS_DIR}) evaluation_latest

echo "✅ Results copied successfully!"
echo "  Latest results: ${PROJECT_ROOT}/results/evaluation_latest -> $(basename ${PERMANENT_RESULTS_DIR})"

# Display storage usage summary
echo "📊 Storage usage summary:"
echo "  Fast storage used: $(du -sh ${EVAL_DIR} | cut -f1)"
echo "  Final results size: $(du -sh ${PERMANENT_RESULTS_DIR} | cut -f1)"

# Display encoder results if available
echo "📈 Encoder Results Summary:"
if [[ -f "${PERMANENT_RESULTS_DIR}/encoder_eval/evaluation_results.json" ]]; then
    echo "  Encoder Model Results:"
    python -c "
import json
with open('${PERMANENT_RESULTS_DIR}/encoder_eval/evaluation_results.json', 'r') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    print(f'    Accuracy: {metrics.get(\"accuracy\", 0):.4f}')
    print(f'    F1-Score: {metrics.get(\"f1\", 0):.4f}')
    print(f'    Precision: {metrics.get(\"precision\", 0):.4f}')
    print(f'    Recall: {metrics.get(\"recall\", 0):.4f}')
" 2>/dev/null || echo "    Results file found but could not parse"
fi

echo "========================================="
echo "✅ Batch Encoder Evaluation Completed Successfully!"
echo "End Time: $(date)"
echo "Job Duration: $((SECONDS / 60)) minutes"
echo "Results saved to: ${PERMANENT_RESULTS_DIR}"
echo "========================================="

# Note: Fast local storage cleanup is automatic
echo "🧹 Fast local storage will be automatically cleaned when job completes"
