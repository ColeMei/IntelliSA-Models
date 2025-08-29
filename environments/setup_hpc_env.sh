#!/bin/bash
# HPC Environment Setup Script for Spartan - Optimized for local NVMe storage

echo "🔧 Setting up HPC environment with fast local storage..."

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
export FAST_STORAGE="/tmp"  # Use fast local NVMe storage

# Cache directories (now on fast local storage)
export HF_HOME="${FAST_STORAGE}/model_cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${FAST_STORAGE}/model_cache/torch_cache"
export WANDB_CACHE_DIR="${FAST_STORAGE}/model_cache/wandb_cache"

# Create cache directories on fast storage
mkdir -p ${HF_HUB_CACHE} ${HF_DATASETS_CACHE} ${TORCH_HOME} ${WANDB_CACHE_DIR}

# GPU settings - let SLURM handle GPU allocation
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "🎯 SLURM Job detected - GPU allocation:"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "  SLURM_GPUS: ${SLURM_GPUS}"
    echo "  SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE}"
    echo "  Fast storage available: $(df -h /tmp | tail -1 | awk '{print $4}')"
fi

# Virtual environment setup - use fast local storage for SLURM jobs
if [ ! -z "$SLURM_JOB_ID" ]; then
    # Use fast local storage for job-specific venv
    VENV_PATH="${FAST_STORAGE}/venv_${SLURM_JOB_ID}"
    echo "🚀 Creating virtual environment on fast storage: ${VENV_PATH}..."
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
    
    echo "✅ Virtual environment created on fast storage"
else
    # Use existing local venv for non-SLURM usage
    if [ -d "${PROJECT_ROOT}/environments/venv" ]; then
        echo "🔄 Using existing virtual environment..."
        source ${PROJECT_ROOT}/environments/venv/bin/activate
    else
        echo "❌ No virtual environment found for local usage"
        exit 1
    fi
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
        rm -rf ${VENV_PATH} 2>/dev/null || true
    }
    trap cleanup_venv EXIT
fi

echo "✅ HPC environment ready!"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Fast storage: ${FAST_STORAGE}"
echo "  Python path: ${PYTHONPATH}"
echo "  Cache dirs: ${HF_HUB_CACHE}, ${HF_DATASETS_CACHE}, ${TORCH_HOME}, ${WANDB_CACHE_DIR}"