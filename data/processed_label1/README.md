# Label Variant 1 Dataset

This directory contains the first label variant for the label quality comparison experiment.

## Structure

```
processed_label1/
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

**Labeler**: Claude Sonnet 4.0 (claude-sonnet-4-0)

This is the baseline labeling strategy used in the main experiment. Labels are generated using Claude Sonnet 4.0 as the post-filter for GLITCH static analysis detections, distinguishing true positives from false positives.

## Usage

This dataset is used with the champion model configuration to compare label quality impact on model performance.

```bash
# Train champion model on this label variant
sbatch scripts/slurm/label_comparison_training.slurm processed_label1
```
