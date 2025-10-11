# Label Variant 3 Dataset

This directory contains the third label variant for the label quality comparison experiment.

## Structure

```
processed_label3/
├── train.jsonl          # Combined training data (all IaC technologies)
├── val.jsonl            # Combined validation data
├── ansible/
│   └── ansible_test.jsonl
├── chef/
│   └── chef_test.jsonl
└── puppet/
    └── puppet_test.jsonl
```

## Labeling Strategy

**Labeler**: GPT-5 (gpt-5-2025-08-07)

Labels are generated using GPT-5 as an alternative labeling strategy for comparison with the baseline Claude Sonnet 4.0 approach.

## Usage

This dataset is used with the champion model configuration to compare label quality impact on model performance.

```bash
# Train champion model on this label variant
sbatch scripts/slurm/label_comparison_training.slurm processed_label3
```
