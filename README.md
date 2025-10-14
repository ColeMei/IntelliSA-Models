# IntelliSA-Models

> **Project Hub**: [IntelliSA](../00.IntelliSA)  
> **This repository**: Systematic model training pipeline  
> See the hub for paper materials, artifact manifest, and links to all repositories.

## Overview

This repository implements a 4-stage systematic training pipeline for encoder models that power IntelliSA's neural inference component. These models reduce false positives in IaC security analysis across Ansible, Chef, and Puppet.

**Problem**: Rule-based static analysis tools generate high false positives across different IaC technologies.

**Solution**: Train encoder LLMs to perform neural inference on rule-based detections, filtering false positives while preserving true vulnerabilities.

**Training Strategy**: Combined training on all IaC technologies with single threshold optimization.

## Training Pipeline

### Stage 1: Broad Candidate Selection

- **Models**: CodeBERT, CodeT5 (small/base), CodeT5+ (220M), UniXcoder
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
│   ├── trainers/                 # Training implementations
│   ├── evaluation/               # Model evaluation
│   └── utils/                    # Utilities
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

## Champion Model

The selected champion model (CodeT5p-220M) is integrated into the `iacsec` CLI tool as IntelliSA's neural inference component.

## Acknowledgements

This research was supported by The University of Melbourne’s Research Computing Services and the Petascale Campus Initiative.
