# LLM-IaC-SecEval-Models

> **Project Hub**: [LLM-IaC-SecEval](../00.LLM-IaC-SecEval)  
> **This repository**: 4-stage systematic model training pipeline  
> See the hub for paper materials, artifact manifest, and links to all repositories.

## Overview

This repository implements a 4-stage systematic training pipeline for encoder models to reduce false positives in IaC security analysis across Ansible, Chef, and Puppet.

**Problem**: Static analysis tools generate high false positives across different IaC technologies.

**Solution**: Use encoder LLMs to filter false positives, making static analysis more usable.

**Training Strategy**: Combined training on all IaC technologies with single threshold optimization.

## Training Pipeline

### Stage 1: Broad Candidate Selection
- **Models**: CodeBERT, CodeT5 (small/base), CodeT5+ (220M)
- **Purpose**: Identify promising model families
- **Evaluation**: Argmax decision rule

### Stage 2: Focused Hyperparameter Tuning
- **Models**: CodeT5+ (220M, 770M) - top performers
- **Purpose**: Optimize learning rates and batch sizes
- **Evaluation**: Argmax decision rule

### Stage 3: Final Optimization with Threshold Sweep
- **Models**: CodeT5+ (220M) - best configuration
- **Purpose**: Threshold optimization and final tuning
- **Evaluation**: Single threshold optimized on validation set

### Stage 4: Multi-Seed Stability Testing
- **Models**: CodeT5+ (220M) - champion configuration
- **Purpose**: Multi-seed stability validation
- **Evaluation**: Single threshold frozen from Stage 3

## Repository Structure

```
├── src/
│   ├── trainers/                  # Training implementations
│   ├── evaluation/                # Model evaluation
│   └── utils/                     # Utilities
├── scripts/
│   ├── batch_train_models.py     # Batch training (HPC)
│   ├── batch_evaluate_models.py  # Batch evaluation (HPC)
│   └── slurm/                    # SLURM job scripts
├── configs/
│   ├── encoder/                  # Training configs (Stage 1-4)
│   ├── eval/                     # Evaluation configs
│   └── champion/                 # Champion configs (Stage 4)
├── local/
│   ├── evaluation/               # Results analysis
│   └── champion_check/           # Champion selection
├── data/                         # Datasets (not synced)
├── models/                       # Trained models (not synced)
└── results/                      # Experimental results
```

## Quick Start

### Training Pipeline

```bash
# Stage 1: Broad sweep
python scripts/batch_train_models.py --config configs/encoder/stage1_batch_training.yaml

# Stage 2: Focused tuning
python scripts/batch_train_models.py --config configs/encoder/stage2_batch_training_220m_770m.yaml

# Stage 3: Final optimization
python scripts/batch_train_models.py --config configs/encoder/stage3_final_sweep_220m.yaml

# Stage 4: Multi-seed champion training
python scripts/batch_train_models.py --config configs/champion/stage4_champion_codet5p220m.yaml
```

### Evaluation Pipeline

```bash
# Stage 1-2: Argmax evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_argmax.yaml

# Stage 3-4: Threshold-based evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_threshold.yaml
```

### Analysis

```bash
# Stage-specific analysis
python local/evaluation/scripts/analyze_evaluation_results.py --prefix codet5p_220m_champion_

# Champion selection
python local/champion_check/select_champion.py --select-only
python local/champion_check/select_champion.py --freeze-only
```

## Key Features

**Single Threshold Design**:
- Training: Combined dataset (all IaC technologies)
- Optimization: Validation set threshold sweep (F1 score)
- Application: Same threshold for all test sets
- Range: 0.3-0.7 with 0.01 step size

**Model Selection Strategy**:
- Stage 1-2: Model family comparison
- Stage 3: Hyperparameter optimization
- Stage 4: Multi-seed stability (median F1 selection)

**Resource Management**:
- Early stopping: 2-step patience, 0.001 F1 threshold
- Mixed precision: FP16 training
- Gradient checkpointing: Large models only
- Batch sizing: Size-appropriate batches (4-32)

## Success Metrics

- Primary: F1 score maximization
- Secondary: False positive reduction rate
- Tertiary: Training efficiency and resource utilization
