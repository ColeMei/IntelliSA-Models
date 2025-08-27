#!/bin/bash
# HPC Environment Setup Script for Spartan (Project Storage Only)

echo "🔧 Setting up HPC environment..."

# Load required modules
module purge
module load Python/3.9.16-GCCcore-12.3.0
module load CUDA/12.1.1
module load GCC/12.3.0

# Set environment variables
export PROJECT_ID="punim2518"
export PROJECT_ROOT="/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models"
export TEMP_ROOT="/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/temp_storage"
export TRANSFORMERS_CACHE="${TEMP_ROOT}/model_cache/huggingface_cache"
export HF_HOME="${TEMP_ROOT}/model_cache/huggingface_cache"
export HF_DATASETS_CACHE="${TEMP_ROOT}/model_cache/huggingface_cache"
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}

# Create Python virtual environment if it doesn't exist
if [ ! -d "${PROJECT_ROOT}/environments/venv" ]; then
    echo "Creating virtual environment..."
    cd ${PROJECT_ROOT}/environments
    python -m venv venv
fi

# Activate virtual environment
source ${PROJECT_ROOT}/environments/venv/bin/activate

# Install requirements if not already installed
if [ ! -f "${PROJECT_ROOT}/environments/.installed" ]; then
    echo "Installing Python packages..."
    pip install --upgrade pip
    pip install -r ${PROJECT_ROOT}/requirements_hpc.txt
    touch ${PROJECT_ROOT}/environments/.installed
    echo "✅ Packages installed"
fi

echo "✅ HPC environment ready!"
echo "Project root: ${PROJECT_ROOT}"
echo "Temp storage: ${TEMP_ROOT}"
echo "Python: $(which python)"
echo "CUDA: ${CUDA_VISIBLE_DEVICES}"
