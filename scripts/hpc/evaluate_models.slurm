#!/bin/bash
#SBATCH --job-name=chef_evaluation
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-4:00:00
#SBATCH --tmp=20GB
#SBATCH --output=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_evaluation.out
#SBATCH --error=/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/logs/slurm_outputs/%j_evaluation.err
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
echo "🗂️ Setting up fast local storage structure..."
FAST_STORAGE="/tmp"
EVAL_DIR="${FAST_STORAGE}/evaluation_${SLURM_JOB_ID}"
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

# Check which models to evaluate based on command line args or available models
APPROACH=${1:-"both"}  # Can be: generative, encoder, or both
GENERATIVE_MODEL_PATH=${2:-"${PROJECT_ROOT}/models/generative_latest"}
ENCODER_MODEL_PATH=${3:-"${PROJECT_ROOT}/models/encoder_latest"}

echo "🎯 Evaluation approach: ${APPROACH}"
echo "🤖 Generative model: ${GENERATIVE_MODEL_PATH}"
echo "🤖 Encoder model: ${ENCODER_MODEL_PATH}"

cd ${PROJECT_ROOT}

# Function to evaluate single model
evaluate_single_model() {
    local approach=$1
    local model_path=$2
    local output_suffix=$3
    
    if [[ ! -d "${model_path}" ]]; then
        echo "⚠️ Model not found: ${model_path}"
        return 1
    fi
    
    echo "🚀 Evaluating ${approach} model..."
    
    srun --unbuffered python scripts/evaluate_models.py \
        --approach ${approach} \
        --model-path ${model_path} \
        --test-path ${DATA_CACHE}/chef_test.jsonl \
        --output-dir ${RESULTS_CACHE}/${approach}_${output_suffix} \
        --batch-size 4 \
        --save-predictions \
        --confusion-matrix 2>&1 | tee ${TEMP_LOGS}/${approach}_evaluation_${SLURM_JOB_ID}.log
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "❌ ${approach} evaluation failed with exit code $exit_code"
        return $exit_code
    fi
    
    echo "✅ ${approach} evaluation completed successfully!"
    return 0
}

# Evaluate models based on approach
EVAL_SUCCESS=0

if [[ "${APPROACH}" == "generative" ]] || [[ "${APPROACH}" == "both" ]]; then
    evaluate_single_model "generative" "${GENERATIVE_MODEL_PATH}" "eval"
    GENERATIVE_EXIT=$?
    if [ $GENERATIVE_EXIT -ne 0 ]; then
        EVAL_SUCCESS=1
    fi
fi

if [[ "${APPROACH}" == "encoder" ]] || [[ "${APPROACH}" == "both" ]]; then
    evaluate_single_model "encoder" "${ENCODER_MODEL_PATH}" "eval"
    ENCODER_EXIT=$?
    if [ $ENCODER_EXIT -ne 0 ]; then
        EVAL_SUCCESS=1
    fi
fi

# If both models were evaluated successfully, run comparison
if [[ "${APPROACH}" == "both" ]] && [ $EVAL_SUCCESS -eq 0 ]; then
    echo "🔄 Running model comparison..."
    
    srun --unbuffered python scripts/evaluate_models.py \
        --approach compare \
        --model-paths ${GENERATIVE_MODEL_PATH} ${ENCODER_MODEL_PATH} \
        --test-path ${DATA_CACHE}/chef_test.jsonl \
        --output-dir ${RESULTS_CACHE}/comparison \
        --save-predictions \
        --confusion-matrix 2>&1 | tee ${TEMP_LOGS}/comparison_${SLURM_JOB_ID}.log
    
    COMPARISON_EXIT=$?
    if [ $COMPARISON_EXIT -ne 0 ]; then
        echo "⚠️ Comparison failed, but individual evaluations succeeded"
    else
        echo "✅ Model comparison completed successfully!"
    fi
fi

# Check overall success
if [ $EVAL_SUCCESS -ne 0 ]; then
    echo "❌ Some evaluations failed"
    # Still copy available results for debugging
    mkdir -p ${PROJECT_ROOT}/logs/evaluation/
    cp -r ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/evaluation/ 2>/dev/null || true
    exit $EVAL_SUCCESS
fi

echo "✅ All evaluations completed successfully!"

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

# Display key results if available
echo "📈 Quick Results Summary:"
if [[ -f "${PERMANENT_RESULTS_DIR}/generative_eval/evaluation_results.json" ]]; then
    echo "  Generative Model Results:"
    python -c "
import json
with open('${PERMANENT_RESULTS_DIR}/generative_eval/evaluation_results.json', 'r') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    print(f'    Accuracy: {metrics.get(\"accuracy\", 0):.4f}')
    print(f'    F1-Score: {metrics.get(\"f1\", 0):.4f}')
    print(f'    Precision: {metrics.get(\"precision\", 0):.4f}')
    print(f'    Recall: {metrics.get(\"recall\", 0):.4f}')
" 2>/dev/null || echo "    Results file found but could not parse"
fi

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

if [[ -f "${PERMANENT_RESULTS_DIR}/comparison/model_comparison.json" ]]; then
    echo "  Best Models:"
    python -c "
import json
with open('${PERMANENT_RESULTS_DIR}/comparison/model_comparison.json', 'r') as f:
    data = json.load(f)
    best = data.get('best_model', {})
    for metric, model in best.items():
        print(f'    Best {metric.replace(\"_\", \" \").title()}: {model}')
" 2>/dev/null || echo "    Comparison file found but could not parse"
fi

echo "========================================="
echo "✅ Chef Detection Model Evaluation Completed Successfully!"
echo "End Time: $(date)"
echo "Job Duration: $((SECONDS / 60)) minutes"
echo "Results saved to: ${PERMANENT_RESULTS_DIR}"
echo "========================================="

# Note: Fast local storage cleanup is automatic
echo "🧹 Fast local storage will be automatically cleaned when job completes"