# LLM-IaC-SecEval-Models

**4-Stage Training Pipeline for Encoder Models to Reduce IaC Security False Positives**

This repository implements a systematic training pipeline for encoder models to reduce false positives in IaC security analysis across Chef, Ansible, and Puppet.

## 🎯 Project Overview

**Problem**: Static analysis tools (even SOTA like GLITCH) generate high false positives (FP) across different IaC technologies.

**Solution**: Use encoder LLMs to filter/reduce FP → making static analysis more usable across Chef, Ansible, and Puppet.

**Approach**: 4-stage systematic training pipeline:

- **Stage 1**: Broad candidate selection (multiple model families)
- **Stage 2**: Focused hyperparameter tuning (top performers)
- **Stage 3**: Final optimization with threshold sweep
- **Stage 4**: Multi-seed stability testing

**Training Strategy**: Combined training on all IaC technologies with single threshold optimization.

## 📁 Repository Structure

```
llm-iac-seceval-models/
├── src/                           # Source code
│   ├── trainers/                  # Training implementations
│   ├── evaluation/                # Model evaluation
│   └── utils/                     # Utilities
├── scripts/                       # Training and evaluation scripts
│   ├── batch_train_models.py     # Batch training (HPC)
│   ├── batch_evaluate_models.py  # Batch evaluation (HPC)
│   └── slurm/                    # SLURM job scripts
├── configs/                       # Stage-specific configurations
│   ├── encoder/                  # Training configs (Stage 1-4)
│   ├── eval/                     # Evaluation configs
│   └── champion/                 # Champion configs (Stage 4)
├── local/                         # Local analysis scripts
│   ├── evaluation/               # Results analysis
│   └── champion_check/           # Champion selection
├── docs/                          # Documentation
├── data/                         # Datasets (not synced)
├── models/                       # Trained models (not synced)
├── results/                      # Experimental results
└── tests/                        # Unit tests
```

## 🔬 Training Pipeline

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
- **Purpose**: Threshold optimization + final tuning
- **Evaluation**: Single threshold (optimized on validation)

### Stage 4: Multi-Seed Stability Testing
- **Models**: CodeT5+ (220M) - champion configuration
- **Purpose**: Multi-seed stability validation
- **Evaluation**: Single threshold (frozen from Stage 3)

## 🚀 Quick Start

### Training Pipeline

```bash
# Stage 1: Broad sweep
python scripts/batch_train_models.py --config configs/encoder/stage1_batch_training.yaml

# Stage 2: Focused tuning
python scripts/batch_train_models.py --config configs/encoder/stage2_batch_training_220m_770m.yaml

# Stage 3: Final optimization with threshold sweep
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

### Analysis Pipeline

```bash
# Stage-specific analysis
python local/evaluation/scripts/analyze_evaluation_results.py --prefix codet5p_220m_champion_

# Champion selection and freezing
python local/champion_check/select_champion.py --select-only
python local/champion_check/select_champion.py --freeze-only
```

## 📊 Key Features

### Single Threshold Design
- **Training**: Combined dataset (all IaC technologies)
- **Optimization**: Validation set threshold sweep (F1 score)
- **Application**: Same threshold for all test sets
- **Range**: 0.3-0.7 with 0.01 step size

### Model Selection Strategy
- **Stage 1-2**: Model family comparison
- **Stage 3**: Hyperparameter optimization
- **Stage 4**: Multi-seed stability (median F1 selection)

### Resource Management
- **Early stopping**: 2-step patience, 0.001 F1 threshold
- **Mixed precision**: FP16 training for memory efficiency
- **Gradient checkpointing**: Large models only
- **Batch sizing**: Size-appropriate batches (4-32)

## 📈 Success Metrics

- **Primary**: F1 score maximization
- **Secondary**: False positive reduction rate
- **Tertiary**: Training efficiency and resource utilization

## 📚 References

- **Main Project**: [llm-iac-seceval](https://github.com/colemei/llm-iac-seceval)
- **GLITCH**: Static analysis tool for IaC security

## 📄 License

This project is part of academic research at University of Melbourne. See LICENSE file for details.
