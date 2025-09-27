# Evaluation Strategy

## Overview

4-stage evaluation pipeline aligned with training stages, using single threshold optimization for encoder models.

## Evaluation Stages

### Stage 1-2: Argmax Evaluation
- **Decision Rule**: Argmax (highest probability class)
- **Purpose**: Model family comparison and hyperparameter tuning
- **Config**: `configs/eval/eval_argmax.yaml`

### Stage 3-4: Threshold-Based Evaluation
- **Decision Rule**: Single threshold (optimized on validation set)
- **Purpose**: Final optimization and stability testing
- **Config**: `configs/eval/eval_threshold.yaml`

### Stage 4 Freeze: Champion Lock-In
- **Decision Rule**: Single threshold reused from Stage 3 champion
- **Purpose**: Produce reproducible champion metrics without manual path edits
- **Config**: `configs/eval/eval_threshold_frozen.yaml`

## Threshold Strategy

### Single Threshold Design
- **Training**: Combined dataset (all IaC technologies)
- **Optimization**: Validation set threshold sweep (F1 score)
- **Application**: Same threshold for all test sets
- **Range**: 0.3-0.7 with 0.01 step size

### Threshold Files
- **Training**: `threshold_sweep_results.json` per model
- **Evaluation**: Automatic loading from model directory
- **Champion freeze**: `codet5p_220m_final_sweep_latest/threshold_sweep_results.json` is discovered via symlink in Stage 4 frozen config
- **Format**: `{"best_threshold": 0.45, "best_score": 0.82, "metric": "f1"}`

## Batch Evaluation Commands

```bash
# Stage 1-2: Argmax evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_argmax.yaml

# Stage 3-4: Threshold-based evaluation
python scripts/batch_evaluate_models.py --config configs/eval/eval_threshold.yaml

# Filter specific models
python scripts/batch_evaluate_models.py --prefix codet5p_220m_champion_ --config configs/eval/eval_threshold.yaml
```

## Test Sets

- **Combined**: `data/processed/test.jsonl` (all technologies)
- **Chef**: `data/processed/chef/chef_test.jsonl`
- **Ansible**: `data/processed/ansible/ansible_test.jsonl`
- **Puppet**: `data/processed/puppet/puppet_test.jsonl`

## Output Structure

```
results/experiments/evaluation/
├── codet5p_220m_champion_lr4e-5_bs8_ep6_wd0.01_acc1_seed41_20250921_221815_job15951614/
│   ├── combined/
│   │   ├── encoder_eval/evaluation_results.json
│   │   └── evaluation_metadata.json
│   ├── chef/
│   │   ├── encoder_eval/evaluation_results.json
│   │   └── evaluation_metadata.json
│   ├── ansible/
│   │   ├── encoder_eval/evaluation_results.json
│   │   └── evaluation_metadata.json
│   └── puppet/
│       ├── encoder_eval/evaluation_results.json
│       └── evaluation_metadata.json
└── ...
```

## Analysis Pipeline

```bash
# Stage-specific analysis
python local/evaluation/scripts/analyze_evaluation_results.py --prefix codet5p_220m_champion_

# All results analysis
python local/evaluation/scripts/analyze_evaluation_results.py

# Export detailed results
python local/evaluation/scripts/analyze_evaluation_results.py --export-csv --export-dataset-csvs
```

## Key Metrics

- **Primary**: F1 score (harmonic mean of precision and recall)
- **Secondary**: Accuracy, Precision, Recall
- **Per-technology**: Individual performance on Chef/Ansible/Puppet
- **Stability**: Multi-seed variance analysis (Stage 4)
