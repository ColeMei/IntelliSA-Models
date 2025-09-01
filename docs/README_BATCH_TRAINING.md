# Batch Training System

This system allows you to run batch training experiments for encoder models with different hyperparameters.

## Quick Start

```bash
# Run complete workflow (training + evaluation)
python scripts/run_batch_experiment.py

# Dry run to see what would be executed
python scripts/run_batch_experiment.py --dry-run

# Run only training
python scripts/run_batch_experiment.py --phases training

# Run only evaluation
python scripts/run_batch_experiment.py --phases evaluation

# Limit experiments
python scripts/run_batch_experiment.py --max-experiments 10

# Filter specific models
python scripts/run_batch_experiment.py --filter codebert
```

## Configuration

Edit `configs/encoder/batch_training_config.yaml` to modify:

- Model hyperparameters (learning rates, batch sizes, epochs)
- SLURM job settings
- Output directories

## Directory Structure

```
configs/
├── encoder/
│   └── batch_training_config.yaml
└── generative/
    └── batch_training_config.yaml (placeholder)

models/experiments/encoder/
├── codebert_base_lr2e-5_bs8_ep3_20241201_120000_job12345/
└── ...

results/experiments/encoder/
├── codebert_base_lr2e-5_bs8_ep3_20241201_120000_job12345/
└── ...

configs/experiments/encoder/
├── codebert_base_lr2e-5_bs8_ep3.yaml
└── ...
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/slurm_outputs/batch_encoder_*.out

# Check job tracking
cat logs/batch_training_jobs.txt
cat logs/batch_evaluation_jobs.txt
```
