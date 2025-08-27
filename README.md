# LLM-IaC-SecEval-Models

**Training Pipeline for LLM-based IaC Security Evaluation Models**

This repository contains the training infrastructure for Stage 2 & Stage 3 of the LLM-IaC-SecEval research project, focusing on training and evaluating models to reduce false positives in static analysis tools.

## 🎯 Project Overview

**Problem**: Static analysis tools (even SOTA like GLITCH) generate high false positives (FP).

**Solution**: Use LLM power to filter/reduce FP → making static analysis more usable.

**Approach**: Compare two training strategies:

- **Generative LLMs** (CodeLLaMA + LoRA): Context-aware reasoning with prompts
- **Encoder-only LLMs** (CodeBERT/CodeT5): Efficient binary classification

## 📁 Repository Structure

```
llm-iac-seceval-models/
├── src/                           # Source code
│   ├── trainers/                  # Training implementations
│   ├── datasets/                  # Dataset classes
│   ├── evaluation/                # Model evaluation
│   └── utils/                     # Utilities
├── scripts/                       # Training and utility scripts
│   ├── train_models.py           # Main training script
│   ├── evaluate_models.py        # Model evaluation
│   └── hpc/                      # HPC-specific scripts
├── configs/                       # Model configurations
├── data/                         # Datasets (not synced)
├── models/                       # Trained models (not synced)
├── logs/                         # Training logs (not synced)
├── results/                      # Experimental results
├── tests/                        # Unit and integration tests
└── environments/                 # Environment setup
```

## 🔬 Research Pipeline

### Stage 2: Model Training Approaches

**Encoder Approach (CodeBERT)**

- Input: Raw code snippets
- Task: Binary classification (TP/FP)
- Architecture: Encoder + classification head
- Training: Full fine-tuning

**Generative Approach (CodeLLaMA)**

- Input: Formatted prompts with context
- Task: Generate "TP" or "FP" responses
- Architecture: Causal LM + LoRA adapters
- Training: Parameter-efficient fine-tuning

### Stage 3: Evaluation & Comparison

- Performance metrics: Precision, Recall, F1, FP reduction rate
- Cost-benefit analysis: Local fine-tuned vs. API calls
- Scaling study: 7B → 13B → 32B model comparison

## 📊 Model Configurations

### Encoder Models

- **CodeBERT**: `microsoft/codebert-base`
- **CodeT5**: `Salesforce/codet5-base`
- Max length: 256 tokens
- Batch size: 8
- Learning rate: 2e-5

### Generative Models

- **CodeLLaMA-7B**: `codellama/CodeLlama-7b-hf`
- **CodeLLaMA-13B**: `codellama/CodeLlama-13b-hf`
- **CodeLLaMA-32B**: `codellama/CodeLlama-32b-hf`
- Max length: 512 tokens
- LoRA rank: 16, alpha: 32
- Batch size: 1-2
- Learning rate: 5e-5

## 📈 Performance Monitoring

### Training Metrics

- Loss curves (training/validation)
- Accuracy, Precision, Recall, F1
- Learning rate schedules
- GPU memory usage

### Logging

- **Local**: `logs/local/`
- **HPC**: `logs/slurm_outputs/`
- **TensorBoard**: Automatic logging enabled
- **Weights & Biases**: Optional integration

## 📊 Expected Results

### Model Comparison Matrix

| Model         | Size | Training Time | Memory Usage | F1 Score | FP Reduction |
| ------------- | ---- | ------------- | ------------ | -------- | ------------ |
| CodeBERT      | 125M | ~2 hours      | 16GB         | TBD      | TBD          |
| CodeT5        | 220M | ~3 hours      | 24GB         | TBD      | TBD          |
| CodeLLaMA-7B  | 7B   | ~8 hours      | 32GB         | TBD      | TBD          |
| CodeLLaMA-13B | 13B  | ~16 hours     | 64GB         | TBD      | TBD          |
| CodeLLaMA-32B | 32B  | ~36 hours     | 128GB        | TBD      | TBD          |

### Research Questions

1. **Performance**: Does generative reasoning outperform encoder classification?
2. **Efficiency**: What's the optimal model size for deployment?

## 📚 References

- **Main Project**: [llm-iac-seceval](https://github.com/colemei/llm-iac-seceval)
- **GLITCH**: Static analysis tool for IaC security

## 📄 License

This project is part of academic research at University of Melbourne. See LICENSE file for details.
