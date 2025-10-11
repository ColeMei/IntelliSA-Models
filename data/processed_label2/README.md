# Label Variant 2 Dataset

This directory contains the second label variant for the label quality comparison experiment.

## Structure

```
processed_label2/
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

**Labeler**: Grok 4 Fast (x-ai/grok-4-fast)

Labels are generated using Grok 4 Fast as an alternative labeling strategy for comparison with the baseline Claude Sonnet 4.0 approach.

## Usage

This dataset is used with the champion model configuration to compare label quality impact on model performance.

```bash
# Train champion model on this label variant
sbatch scripts/slurm/label_comparison_training.slurm processed_label2
```
