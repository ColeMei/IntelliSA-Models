# Config stages and evaluation pairing

We use a 4-stage process. Pair each training stage with its appropriate evaluation config.

- Stage 1: Broad sweep (candidate selection)

  - Training: `configs/encoder/stage1_batch_training.yaml`
  - Evaluation: `configs/eval/eval_argmax.yaml` (argmax; no thresholds)

- Stage 2: Focused tuning (LR exploration)

  - Training: `configs/encoder/stage2_batch_training_220m_770m.yaml`
  - Evaluation: `configs/eval/eval_argmax.yaml` (argmax)

- Stage 3: Final focused sweep (single threshold calibrated per run)

  - Training: `configs/encoder/stage3_final_sweep_220m.yaml`
  - Evaluation (single threshold): `configs/eval/eval_threshold.yaml` with
    - `threshold.mode: file` → batch job defaults to each run's `models/experiments/encoder/{run}/threshold_sweep_results.json`
    - Single threshold used for all test sets (combined, chef, ansible, puppet)

- Stage 4: Stability (multi-seed) and champion selection
  - Training: `configs/champion/stage4_champion_codet5p220m.yaml`
  - Evaluation (frozen thresholds): Same as Stage 3

Notes

- Evaluator supports env-driven thresholds (argmax|fixed|file). Batch jobs export the env from the eval config.
- Single threshold design: Training on combined data → single optimal threshold → applied to all test sets.
- Freeze step writes `artifacts/models/champion/*` and `results/experiments/evaluation/frozen_thresholds.yaml`.
