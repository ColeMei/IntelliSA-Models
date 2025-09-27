# Batch Encoder Training Pipeline

## Overview

4-stage systematic training pipeline for encoder models to reduce false positives in IaC security analysis.

## Pipeline Stages

### Stage 1: Broad Candidate Selection
- **Models**: CodeBERT, CodeT5 (small/base), CodeT5+ (220M)
- **Purpose**: Identify promising model families
- **Evaluation**: Argmax decision rule
- **Config**: `configs/encoder/stage1_batch_training.yaml`

### Stage 2: Focused Hyperparameter Tuning  
- **Models**: CodeT5+ (220M, 770M) - top performers from Stage 1
- **Purpose**: Optimize learning rates and batch sizes
- **Evaluation**: Argmax decision rule
- **Config**: `configs/encoder/stage2_batch_training_220m_770m.yaml`

### Stage 3: Final Optimization with Threshold Sweep
- **Models**: CodeT5+ (220M) - best configuration from Stage 2
- **Purpose**: Threshold optimization + final hyperparameter tuning
- **Evaluation**: Single threshold (optimized on validation set)
- **Config**: `configs/encoder/stage3_final_sweep_220m.yaml`

### Stage 4: Multi-Seed Stability Testing
- **Models**: CodeT5+ (220M) - champion configuration from Stage 3
- **Purpose**: Multi-seed stability validation
- **Evaluation**: Single threshold (frozen from Stage 3)
- **Config**: `configs/encoder/stage4_champion_codet5p220m.yaml`

## Training Commands

```bash
# Stage 1: Broad sweep
python scripts/batch_train_models.py --config configs/encoder/stage1_batch_training.yaml

# Stage 2: Focused tuning
python scripts/batch_train_models.py --config configs/encoder/stage2_batch_training_220m_770m.yaml

# Stage 3: Final sweep with threshold optimization
python scripts/batch_train_models.py --config configs/encoder/stage3_final_sweep_220m.yaml

# Stage 4: Multi-seed champion training
python scripts/batch_train_models.py --config configs/encoder/stage4_champion_codet5p220m.yaml
```

## Evaluation Commands

```bash
# Stage 1-2: Argmax evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_argmax.yaml

# Stage 3-4: Threshold-based evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_threshold.yaml

# Stage 4 (champion freeze): Frozen-threshold evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_threshold_frozen.yaml
```

## Key Features

### Threshold Optimization (Stages 3-4)
- **Single threshold** for all test sets (combined, chef, ansible, puppet)
- **Validation-based optimization** using F1 score
- **Range**: 0.3-0.7 with 0.01 step size
- **Saved to**: `threshold_sweep_results.json` per model
- **Stage 4 freeze**: `eval_threshold_frozen.yaml` resolves the champion threshold via the `codet5p_220m_final_sweep_latest` symlink, so no manual path edits are needed

### Model Selection Strategy
- **Stage 1-2**: Model family comparison
- **Stage 3**: Hyperparameter optimization
- **Stage 4**: Multi-seed stability (median F1 selection)

### Resource Management
- **Early stopping**: 2-step patience, 0.001 F1 threshold
- **Mixed precision**: FP16 training for memory efficiency
- **Gradient checkpointing**: Large models only
- **Batch sizing**: Size-appropriate batches (4-32)

## Output Structure

```
models/experiments/encoder/
├── codet5p_220m_lr2e-5_bs8_ep3_wd0.01_20250921_123456_job12345/
│   ├── config_used.yaml
│   ├── threshold_sweep_results.json  # Stages 3-4 only
│   └── pytorch_model.bin
└── ...

results/experiments/evaluation/
├── codet5p_220m_lr2e-5_bs8_ep3_wd0.01_20250921_123456_job12345/
│   ├── combined/evaluation_metadata.json
│   ├── chef/evaluation_metadata.json
│   ├── ansible/evaluation_metadata.json
│   └── puppet/evaluation_metadata.json
└── ...
```

## Analysis Pipeline

```bash
# Stage-specific analysis
python local/evaluation/scripts/analyze_evaluation_results.py --prefix codet5p_220m_champion_

# Champion selection and freezing
python local/champion_check/select_champion.py --select-only
python local/champion_check/select_champion.py --freeze-only
```

## Success Metrics

- **Primary**: F1 score maximization
- **Secondary**: False positive reduction rate
- **Tertiary**: Training efficiency and resource utilization
