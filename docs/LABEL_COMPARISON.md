# Label Comparison Experiment

## Overview

This experiment validates the quality of our labeling strategy by training the champion model configuration on different label variants while holding all other factors constant.

## Research Question

**Does our labeling approach (GLITCH + LLM post-filter pseudo-labels) produce superior model performance compared to alternative labeling strategies?**

## Experiment Design

### Controlled Variables (Frozen)
- **Model Architecture**: CodeT5+ 220M
- **Hyperparameters**: Learning rate 4e-5, batch size 8, 6 epochs, weight decay 0.01
- **Training Strategy**: Combined training on all IaC technologies
- **Test Sets**: Identical test sets (combined/chef/ansible/puppet)
- **Evaluation**: Same threshold sweep strategy (F1-optimized)

### Independent Variable (Varied)
- **Training Labels**: 3 different labeling strategies on identical IaC samples
  - `processed_label1`: [Document labeling strategy]
  - `processed_label2`: [Document labeling strategy]
  - `processed_label3`: [Document labeling strategy]

### Dependent Variables (Measured)
- **Primary**: F1 score on test sets
- **Secondary**: Precision, Recall, Accuracy
- **Per-technology**: Performance on Chef/Ansible/Puppet individually

## Directory Structure

```
data/
├── processed/                 # Original dataset (baseline reference)
├── processed_label1/          # Label variant 1
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── ansible/ansible_test.jsonl
│   ├── chef/chef_test.jsonl
│   └── puppet/puppet_test.jsonl
├── processed_label2/          # Label variant 2
│   └── (same structure)
└── processed_label3/          # Label variant 3
    └── (same structure)

models/experiments/label_comparison/
├── codet5p_220m_label_comparison_processed_label1_seed41_*/
├── codet5p_220m_label_comparison_processed_label1_seed42_*/
├── codet5p_220m_label_comparison_processed_label1_seed43_*/
├── codet5p_220m_label_comparison_processed_label2_seed41_*/
└── ...

results/experiments/label_comparison/
└── (evaluation results per model)
```

## Workflow

### Step 1: Prepare Label Variants

Each label variant should have identical IaC samples with different labels:

```bash
# Verify data structure
for label in label1 label2 label3; do
    echo "Checking processed_${label}..."
    ls -lh data/processed_${label}/
    wc -l data/processed_${label}/*.jsonl
done
```

**Important**: Ensure all variants have:
- Same number of samples in train/val
- Same IaC code snippets
- Only labels differ (TP/FP classifications)

### Step 2: Train Models on All Label Variants

```bash
# Dry run to preview experiments
python scripts/batch_train_label_comparison.py --dry-run

# Submit all training jobs (3 datasets × 3 seeds = 9 jobs)
python scripts/batch_train_label_comparison.py

# Submit specific label variants only
python scripts/batch_train_label_comparison.py --datasets processed_label1 processed_label2

# Limit number of jobs for testing
python scripts/batch_train_label_comparison.py --max-jobs 3
```

This will:
1. Generate individual configs for each experiment
2. Submit SLURM jobs with dataset-specific parameters
3. Train champion configuration on each label variant
4. Save models to `models/experiments/label_comparison/`

### Step 3: Evaluate All Models

```bash
# Evaluate all label comparison models
python scripts/batch_evaluate_models.py \
    --config configs/eval/eval_label_comparison.yaml \
    --prefix codet5p_220m_label_comparison

# Evaluate specific label variant
python scripts/batch_evaluate_models.py \
    --config configs/eval/eval_label_comparison.yaml \
    --prefix codet5p_220m_label_comparison_processed_label1
```

This will:
1. Use threshold sweep results from training
2. Evaluate on identical test sets (combined/chef/ansible/puppet)
3. Save results to `results/experiments/evaluation/label_comparison/`

### Step 4: Analyze Results

```bash
# Analyze label comparison results
python local/evaluation/scripts/analyze_evaluation_results.py \
    --prefix codet5p_220m_label_comparison_ \
    --export-csv

# Compare across label variants
python local/evaluation/scripts/compare_label_variants.py
```

Expected output:
- Performance comparison table (F1, Precision, Recall per variant)
- Statistical significance tests across variants
- Best performing label strategy identification

## Configuration Files

### Training Config
- **File**: `configs/encoder/label_comparison_champion.yaml`
- **Purpose**: Champion hyperparameters for all label variants
- **Seeds**: 41, 42, 43 (for stability)

### Evaluation Config
- **File**: `configs/eval/eval_label_comparison.yaml`
- **Purpose**: Consistent evaluation across all variants
- **Test Sets**: Same for all (ensures fair comparison)

### SLURM Script
- **File**: `scripts/slurm/label_comparison_training.slurm`
- **Parameters**: Config file + dataset variant name
- **Resources**: Same as champion training (A100, 4h, 32GB)

## Success Criteria

1. **Completion**: All 9 models trained successfully (3 variants × 3 seeds)
2. **Consistency**: Low variance within each label variant (across seeds)
3. **Discrimination**: Measurable performance differences between variants
4. **Validation**: Best variant should match or exceed baseline performance

## Analysis Questions

1. Which labeling strategy produces highest F1 scores?
2. Are differences statistically significant?
3. Does performance ranking hold across all technologies (Chef/Ansible/Puppet)?
4. How much does label quality impact vary across seeds?
5. Can we identify specific characteristics of the best labeling approach?

## Expected Outcomes

- **Hypothesis**: Original labeling strategy (baseline) performs best
- **Evidence**: Quantitative comparison validates label quality impact
- **Publication**: Results support claims about labeling approach effectiveness
- **Future Work**: Insights guide labeling strategy improvements

## Notes

- All models use the **same test sets** from `data/processed/` to ensure fair comparison
- Training data differs (label variants) but evaluation data is identical
- This isolates label quality as the only variable affecting performance
- Multi-seed training (3 seeds) ensures results are stable and reproducible
