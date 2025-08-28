#!/bin/bash
# HPC Environment Setup Script for Spartan - Improved Version

echo "🔧 Setting up HPC environment..."

# Load required modules on Spartan (correct order per documentation)
module purge 2>/dev/null || true

# Load toolchain first (required on new module system)
echo "Loading toolchain..."
module load foss/2022a || {
    echo "⚠️ foss/2022a not available, trying GCCcore/11.3.0..."
    module load GCCcore/11.3.0 || {
        echo "❌ Failed to load base toolchain"
        exit 1
    }
}

# Python (following documentation examples)
echo "Loading Python..."
module load Python/3.10.4 || {
    echo "⚠️ Python/3.10.4 not available, trying alternatives..."
    module load Python/3.11.3 || module load Python/3.9.6 || {
        echo "❌ No compatible Python version found"
        exit 1
    }
}

# CUDA (use documented versions)
echo "Loading CUDA..."
module load CUDA/11.7.0 || module load CUDA/12.4.1 || {
    echo "⚠️ CUDA not available, continuing without GPU support"
}

# cuDNN if available
module load cuDNN/8.4.1.50-CUDA-11.7.0 || module load cuDNN/9.6.0.74-CUDA-12.4.1 || {
    echo "⚠️ cuDNN not available, continuing..."
}

echo "📋 Loaded modules:"
module list

# Set environment variables
export PROJECT_ID="punim2518"
export PROJECT_ROOT="/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models"
export TEMP_ROOT="/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models/temp_storage"

# Cache directories (use temp storage for better I/O)
export TRANSFORMERS_CACHE="${TEMP_ROOT}/model_cache/huggingface_cache"
export HF_HOME="${TEMP_ROOT}/model_cache/huggingface_cache"
export HF_DATASETS_CACHE="${TEMP_ROOT}/model_cache/huggingface_cache"
export TORCH_HOME="${TEMP_ROOT}/model_cache/torch_cache"
export WANDB_CACHE_DIR="${TEMP_ROOT}/model_cache/wandb_cache"

# Create cache directories
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${TORCH_HOME}
mkdir -p ${WANDB_CACHE_DIR}

# GPU settings - let SLURM handle GPU allocation
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "🔍 SLURM Job detected - GPU allocation:"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "  SLURM_GPUS: ${SLURM_GPUS}"
    echo "  SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE}"
fi

# Virtual environment setup - use temp storage for better performance
VENV_PATH="${TEMP_ROOT}/venv_${SLURM_JOB_ID:-local}"

# Create virtual environment if running in SLURM job or if doesn't exist locally
if [ ! -z "$SLURM_JOB_ID" ] || [ ! -d "${PROJECT_ROOT}/environments/venv" ]; then
    echo "🐍 Creating virtual environment at ${VENV_PATH}..."
    python -m venv ${VENV_PATH}
    source ${VENV_PATH}/bin/activate
    
    # Upgrade pip and install requirements
    echo "📦 Installing Python packages..."
    pip install --upgrade pip wheel setuptools
    
    # Check if requirements file exists
    if [ -f "${PROJECT_ROOT}/requirements_hpc.txt" ]; then
        pip install -r ${PROJECT_ROOT}/requirements_hpc.txt
    elif [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
        pip install -r ${PROJECT_ROOT}/requirements.txt
    else
        echo "⚠️ No requirements file found, installing basic packages..."
        pip install torch torchvision torchaudio transformers datasets accelerate evaluate scikit-learn pandas numpy
    fi
    
    echo "✅ Virtual environment created and packages installed"
else
    # Use existing local venv
    echo "🐍 Using existing virtual environment..."
    source ${PROJECT_ROOT}/environments/venv/bin/activate
fi

# Verify installation
echo "🔍 Environment verification:"
echo "  Python: $(which python) ($(python --version))"
echo "  Pip packages: $(pip list | wc -l) packages installed"

# CUDA verification
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "  CUDA: $(python -c 'import torch; print("Available" if torch.cuda.is_available() else "Not available")')"
else
    echo "  GPU: Not available or not allocated"
fi

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"

# Cleanup function for temp venv (only if SLURM job)
if [ ! -z "$SLURM_JOB_ID" ]; then
    cleanup_venv() {
        echo "🧹 Cleaning up temporary virtual environment..."
        rm -rf ${VENV_PATH}
    }
    trap cleanup_venv EXIT
fi

echo "✅ HPC environment ready!"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Temp storage: ${TEMP_ROOT}"
echo "  Python path: ${PYTHONPATH}"
echo "  Cache dirs: ${TRANSFORMERS_CACHE}"