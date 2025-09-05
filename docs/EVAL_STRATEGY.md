# EVAL_STRATEGY.md

## IaC Security Smell Detection Model Evaluation Strategy

A unified evaluation framework for comparing generative (CodeLLaMA) and encoder (CodeBERT) approaches on IaC security smell detection across multiple technologies (Chef, Ansible, Puppet).

## Architecture Overview

```
evaluation/
├── evaluate_models.py          # Main evaluation script
├── generative_evaluator.py     # CodeLLaMA evaluation
├── encoder_evaluator.py        # CodeBERT evaluation
└── model_comparator.py         # Cross-model comparison
```

## Key Components

### 1. GenerativeEvaluator

- Loads fine-tuned CodeLLaMA + LoRA adapters
- Uses instruction prompts
- Extracts TP/FP from generated text
- Handles 4-bit quantization for memory efficiency

### 2. EncoderEvaluator

- Loads fine-tuned CodeBERT classification model
- Uses raw code content (no prompts)
- Binary classification with confidence scores
- Batch processing for efficiency

### 3. ModelComparator

- Statistical comparison across models
- Per-smell performance breakdown
- Generates visualizations and HTML reports
- Model ranking by different metrics

## Usage Instructions

### HPC Deployment (Recommended)

```bash
# Evaluate both models (default)
sbatch scripts/hpc/evaluate_models.slurm

# Single model evaluation
sbatch scripts/hpc/evaluate_models.slurm configs/evaluation_config.yaml generative
sbatch scripts/hpc/evaluate_models.slurm configs/evaluation_config.yaml encoder
```

## Configuration

```yaml
# configs/evaluation_config.yaml
test_path: "data/processed/chef_test.jsonl"
batch_size: 4
save_predictions: true

models:
  generative:
    path: "models/generative_latest"
    batch_size: 1
  encoder:
    path: "models/encoder_latest"
    batch_size: 8

generative_use_4bit: true
max_new_tokens: 10
```

## Expected Output Structure

```
results/evaluation_TIMESTAMP_jobID/
├── generative_eval/
│   ├── evaluation_results.json      # Metrics & summary
│   └── detailed_predictions.json    # Per-sample predictions
├── encoder_eval/
│   ├── evaluation_results.json
│   └── detailed_predictions.json
├── comparison/
│   ├── model_comparison.json        # Cross-model analysis
│   ├── comparison_report.html       # Visual report
│   ├── performance_comparison.png   # Bar charts
│   └── radar_comparison.png         # Radar plot
└── evaluation_latest -> [symlink to latest results]
```

## Key Features

- **YAML-driven**: All configuration in one file
- **Unified interface**: Single command for both models
- **HPC optimized**: Fast local storage, SLURM integration
- **Comprehensive metrics**: Accuracy, F1, precision, recall
- **Visual reports**: HTML summaries, charts, confusion matrices
- **Model comparison**: Side-by-side analysis and rankings

## Results Format

```json
{
  "metrics": {
    "accuracy": 0.8542,
    "precision": 0.8123,
    "recall": 0.7891,
    "f1": 0.8005
  },
  "confusion_matrix": { "tp": 156, "fp": 23, "tn": 201, "fn": 45 },
  "smell_metrics": {
    "hard_coded_secret": { "f1": 0.85, "count": 67 },
    "suspicious_comment": { "f1": 0.72, "count": 43 }
  }
}
```
