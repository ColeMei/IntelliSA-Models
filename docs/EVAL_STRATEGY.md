# EVAL_STRATEGY.md

## Chef Detection Model Evaluation Strategy

A unified evaluation framework for comparing generative (CodeLLaMA) and encoder (CodeBERT) approaches on Chef detection classification.

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
# Evaluate both models
sbatch scripts/hpc/evaluate_models.slurm both

# Single model evaluation
sbatch scripts/hpc/evaluate_models.slurm generative /path/to/model
sbatch scripts/hpc/evaluate_models.slurm encoder /path/to/model
```

### Configuration

```yaml
# configs/evaluation_config.yaml
test_path: "data/processed/chef_test.jsonl"
batch_size: 4
save_predictions: true
confusion_matrix: true
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

### Key Metrics Output

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

## Key Features

### 🚀 **Unified Interface**

- Single command evaluates both model types
- Automatic model detection and loading
- Consistent output format across approaches

### 📊 **Comprehensive Metrics**

- Standard classification metrics (Acc, F1, P, R)
- Per-smell type performance breakdown
- Confusion matrices and confidence scores
- Statistical significance testing

### 🔄 **Model Comparison**

- Side-by-side performance analysis
- Visual comparisons (bar charts, radar plots)
- HTML reports with rankings
- Best model identification per metric

### 💾 **HPC Optimized**

- Fast local storage utilization
- Memory-efficient batch processing
- Proper SLURM resource management
- Automatic result archiving

### 🛡️ **Robust Processing**

- Error handling for failed predictions
- Memory optimization for large models
- Batch size auto-adjustment
- Comprehensive logging

### 📈 **Rich Visualization**

- Performance comparison charts
- Per-smell breakdown plots
- Confusion matrix heatmaps
- Interactive HTML reports

## Quick Start

1. **Setup**: Ensure trained models are in `models/generative_latest` and `models/encoder_latest`
2. **Run**: `sbatch scripts/hpc/evaluate_models.slurm both`
3. **Results**: Check `results/evaluation_latest/` for outputs
4. **Analysis**: Open `comparison/comparison_report.html` for visual summary
