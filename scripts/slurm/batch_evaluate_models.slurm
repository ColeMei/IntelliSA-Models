#!/bin/bash
# SLURM parameters are set by the Python script that submits this job
# No need to specify them here as they will be overridden
# This script is used for batch evaluation of trained models

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

# Get arguments: model path, destination directory, batch size, test set path, and optional threshold params
MODEL_PATH="$1"
DEST_DIR="$2"
BATCH_SIZE="$3"
TEST_PATH="$4"
THRESHOLD_MODE="${5:-}"
THRESHOLD_FILE="${6:-}"
THRESHOLD_FIXED="${7:-}"
THRESHOLD_KEY="${8:-}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ No model path provided! Usage: sbatch batch_evaluate_models.slurm <model_path> <dest_dir> <batch_size> <test_path>"
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    exit 1
fi
if [[ -z "$DEST_DIR" ]]; then
    echo "❌ No destination directory provided! Usage: sbatch batch_evaluate_models.slurm <model_path> <dest_dir> <batch_size> <test_path>"
    exit 1
fi
if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE=8  # Default batch size
fi
if [[ -z "$TEST_PATH" ]]; then
    TEST_PATH="data/processed/test.jsonl"  # Default test path
fi
if [[ ! -f "$TEST_PATH" ]]; then
    echo "❌ Test file not found: $TEST_PATH"
    exit 1
fi

echo "🤖 Model path: $MODEL_PATH"
echo "📁 Destination directory: $DEST_DIR"
echo "🔢 Batch size: $BATCH_SIZE"
echo "📋 Test set: $TEST_PATH"

# Export threshold env if provided
if [[ -n "$THRESHOLD_MODE" ]]; then
    export EVAL_THRESHOLD_MODE="$THRESHOLD_MODE"
fi
if [[ -n "$THRESHOLD_FILE" ]]; then
    export EVAL_THRESHOLD_FILE="$THRESHOLD_FILE"
fi
if [[ -n "$THRESHOLD_FIXED" ]]; then
    export EVAL_THRESHOLD_FIXED="$THRESHOLD_FIXED"
fi
if [[ -n "$THRESHOLD_KEY" ]]; then
    export EVAL_THRESHOLD_KEY="$THRESHOLD_KEY"
fi

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
cp -r ${TEST_PATH} ${DATA_CACHE}/test_data.jsonl

# Verify test data exists
if [[ ! -f "${DATA_CACHE}/test_data.jsonl" ]]; then
    echo "❌ Test data not found on fast storage! Exiting..."
    exit 1
fi

echo "✅ Test data staged successfully"

# Set Python path explicitly
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# Evaluation seed for determinism
export EVAL_SEED=${EVAL_SEED:-42}

cd ${PROJECT_ROOT}

# Run encoder evaluation with direct parameters
echo "🚀 Evaluating encoder model..."
MODEL_NAME=$(basename "$MODEL_PATH")
echo "📊 Evaluating model: $MODEL_NAME"

srun --unbuffered python scripts/evaluate_models.py \
    --approach encoder \
    --model-path ${MODEL_PATH} \
    --test-path ${DATA_CACHE}/test_data.jsonl \
    --output-dir ${RESULTS_CACHE}/encoder_eval \
    --batch-size ${BATCH_SIZE} \
    --save-predictions 2>&1 | tee ${TEMP_LOGS}/encoder_evaluation_${SLURM_JOB_ID}.log

# If threshold mode requires a file, assert it is present to lock results
if [[ "${EVAL_THRESHOLD_MODE}" == "file" ]]; then
    if [[ -z "${EVAL_THRESHOLD_FILE}" || ! -f "${EVAL_THRESHOLD_FILE}" ]]; then
        echo "❌ EVAL_THRESHOLD_MODE=file requires a valid EVAL_THRESHOLD_FILE. Missing: ${EVAL_THRESHOLD_FILE}"
        exit 2
    fi
fi

EVAL_EXIT_CODE=$?
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "❌ Encoder evaluation failed with exit code $EVAL_EXIT_CODE"
    # Still copy logs for debugging
    mkdir -p ${PROJECT_ROOT}/logs/evaluation/
    cp -r ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/evaluation/ 2>/dev/null || true
    exit $EVAL_EXIT_CODE
fi

echo "✅ Encoder evaluation completed successfully!"

# Use provided destination directory for permanent storage
PERMANENT_RESULTS_DIR="$DEST_DIR"
EVALUATION_LOGS_DIR="${PROJECT_ROOT}/logs/evaluation/"

echo "💾 Copying evaluation results to permanent storage..."
echo "  Results destination: ${PERMANENT_RESULTS_DIR}"

# Create directories
mkdir -p ${PERMANENT_RESULTS_DIR}
mkdir -p ${EVALUATION_LOGS_DIR}

# Copy all results to permanent storage
cp -r ${RESULTS_CACHE}/* ${PERMANENT_RESULTS_DIR}/

# Copy logs and metadata
cp ${TEMP_LOGS}/* ${PERMANENT_RESULTS_DIR}/ 2>/dev/null || true
cp ${TEMP_LOGS}/* ${EVALUATION_LOGS_DIR}/ 2>/dev/null || true

# Save evaluation metadata
cat > ${PERMANENT_RESULTS_DIR}/evaluation_metadata.json << EOF
{
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "batch_size": "$BATCH_SIZE",
  "job_id": "$SLURM_JOB_ID",
  "timestamp": "$(date -Iseconds)",
  "evaluation_type": "batch_encoder"
}
EOF

echo "✅ Results copied successfully!"

# Display storage usage summary
echo "📊 Storage usage summary:"
echo "  Fast storage used: $(du -sh ${EVAL_DIR} | cut -f1)"
echo "  Final results size: $(du -sh ${PERMANENT_RESULTS_DIR} | cut -f1)"

# Display encoder results if available
echo "📈 Encoder Results Summary:"
RESULTS_FILE="${PERMANENT_RESULTS_DIR}/encoder_eval/evaluation_results.json"
if [[ -f "$RESULTS_FILE" ]]; then
    echo "  📊 Model: $MODEL_NAME"
    python -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)
        metrics = data.get('metrics', {})
        print(f'    Accuracy: {metrics.get(\"accuracy\", 0):.4f}')
        print(f'    F1-Score: {metrics.get(\"f1\", 0):.4f}')
        print(f'    Precision: {metrics.get(\"precision\", 0):.4f}')
        print(f'    Recall: {metrics.get(\"recall\", 0):.4f}')
        print(f'    Test Samples: {data.get(\"num_samples\", 0)}')
        print(f'    Evaluation Time: {data.get(\"evaluation_time\", 0):.2f}s')
except Exception as e:
    print(f'    Error parsing results: {e}')
" 2>/dev/null || echo "    Results file found but could not parse"
else
    echo "  ⚠️  Results file not found at: $RESULTS_FILE"
fi

echo "========================================="
echo "✅ Batch Model Evaluation Completed Successfully!"
echo "📊 Model: $MODEL_NAME"
echo "🔢 Batch Size: $BATCH_SIZE"
echo "⏱️  Job Duration: $((SECONDS / 60)) minutes"
echo "💾 Results saved to: ${PERMANENT_RESULTS_DIR}"
echo "========================================="

# Note: Evaluation config files are now stored under ${PROJECT_ROOT}/logs/batch_eval_configs for reproducibility

# Note: Fast local storage cleanup is automatic
echo "🧹 Fast local storage will be automatically cleaned when job completes"
