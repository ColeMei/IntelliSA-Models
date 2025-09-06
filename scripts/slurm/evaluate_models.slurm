#!/bin/bash
#SBATCH --job-name=evaluation
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
EVAL_DIR="${FAST_STORAGE}/evaluation_${SLURM_JOB_ID}"
DATA_CACHE="${EVAL_DIR}/data_cache"
RESULTS_CACHE="${EVAL_DIR}/results"
TEMP_LOGS="${EVAL_DIR}/logs"

rm -rf ${EVAL_DIR} 2>/dev/null || true
mkdir -p ${DATA_CACHE} ${RESULTS_CACHE} ${TEMP_LOGS}

# Stage and verify data
echo "📋 Staging test data..."
cp -r ${PROJECT_ROOT}/data/processed/test.jsonl ${DATA_CACHE}/
[ -f "${DATA_CACHE}/test.jsonl" ] || { echo "❌ Test data missing"; exit 1; }
echo "✅ Data staged: $(wc -l < ${DATA_CACHE}/test.jsonl) test samples"

# Setup evaluation parameters
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

CONFIG_FILE=${1:-"${PROJECT_ROOT}/configs/evaluation_config.yaml"}
APPROACH=${2:-"both"}

echo "🎯 Approach: ${APPROACH} | Config: ${CONFIG_FILE}"
[ -f "${CONFIG_FILE}" ] || { echo "❌ Config file not found: ${CONFIG_FILE}"; exit 1; }
cd ${PROJECT_ROOT}

# Evaluate models
EVAL_SUCCESS=0

# Evaluate generative model if requested
if [[ "${APPROACH}" == "generative" ]] || [[ "${APPROACH}" == "both" ]]; then
    echo "🚀 Evaluating generative model..."
    srun --unbuffered python scripts/evaluate_models.py \
        --approach generative \
        --config ${CONFIG_FILE} \
        --test-path ${DATA_CACHE}/test.jsonl \
        --output-dir ${RESULTS_CACHE}/generative_eval 2>&1 | tee ${TEMP_LOGS}/generative_evaluation_${SLURM_JOB_ID}.log

    [ $? -eq 0 ] || { echo "❌ Generative evaluation failed"; EVAL_SUCCESS=1; }
fi

# Evaluate encoder model if requested
if [[ "${APPROACH}" == "encoder" ]] || [[ "${APPROACH}" == "both" ]]; then
    echo "🚀 Evaluating encoder model..."
    srun --unbuffered python scripts/evaluate_models.py \
        --approach encoder \
        --config ${CONFIG_FILE} \
        --test-path ${DATA_CACHE}/test.jsonl \
        --output-dir ${RESULTS_CACHE}/encoder_eval 2>&1 | tee ${TEMP_LOGS}/encoder_evaluation_${SLURM_JOB_ID}.log

    [ $? -eq 0 ] || { echo "❌ Encoder evaluation failed"; EVAL_SUCCESS=1; }
fi

# Run comparison if both models evaluated successfully
if [[ "${APPROACH}" == "both" ]] && [ $EVAL_SUCCESS -eq 0 ]; then
    echo "🔄 Running model comparison..."
    srun --unbuffered python scripts/evaluate_models.py \
        --approach compare \
        --config ${CONFIG_FILE} \
        --test-path ${DATA_CACHE}/test.jsonl \
        --output-dir ${RESULTS_CACHE}/comparison 2>&1 | tee ${TEMP_LOGS}/comparison_${SLURM_JOB_ID}.log

    [ $? -eq 0 ] && echo "✅ Comparison completed" || echo "⚠️ Comparison failed"
fi

[ $EVAL_SUCCESS -eq 0 ] || { echo "❌ Some evaluations failed"; cp -r ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/evaluation/ 2>/dev/null || true; exit $EVAL_SUCCESS; }
echo "✅ All evaluations completed successfully!"

# Store results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERMANENT_RESULTS_DIR="${PROJECT_ROOT}/results/evaluation_${TIMESTAMP}_job${SLURM_JOB_ID}"

mkdir -p ${PERMANENT_RESULTS_DIR} ${PROJECT_ROOT}/logs/evaluation/
cp -r ${RESULTS_CACHE}/* ${PERMANENT_RESULTS_DIR}/
cp ${TEMP_LOGS}/* ${PERMANENT_RESULTS_DIR}/ 2>/dev/null || true
cp ${TEMP_LOGS}/* ${PROJECT_ROOT}/logs/evaluation/ 2>/dev/null || true

cd ${PROJECT_ROOT}/results/
rm -f evaluation_latest 2>/dev/null || true
ln -s $(basename ${PERMANENT_RESULTS_DIR}) evaluation_latest

echo "✅ Results saved to: ${PERMANENT_RESULTS_DIR}"

# Display summary
echo "📊 Storage: $(du -sh ${EVAL_DIR} | cut -f1) used, $(du -sh ${PERMANENT_RESULTS_DIR} | cut -f1) results"

# Quick results summary
echo "📈 Results Summary:"
for model in generative encoder; do
    results_file="${PERMANENT_RESULTS_DIR}/${model}_eval/evaluation_results.json"
    if [[ -f "$results_file" ]]; then
        echo "  ${model^} Model:"
        python -c "
import json
with open('$results_file', 'r') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    print(f'    Acc: {metrics.get(\"accuracy\", 0):.4f}, F1: {metrics.get(\"f1\", 0):.4f}')
" 2>/dev/null || echo "    (Parse error)"
    fi
done

echo "========================================="
echo "✅ Evaluation completed in $((SECONDS / 60)) minutes"
echo "Results: ${PROJECT_ROOT}/results/evaluation_latest"
echo "========================================="

# Note: Fast local storage cleanup is automatic
echo "🧹 Fast local storage will be automatically cleaned when job completes"