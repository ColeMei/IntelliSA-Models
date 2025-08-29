# 📂 LLM-IaC-SecEval-Models – Repository Design

## 1. Core Idea

This repository serves as a **shared codebase** for model training workflows, used across both **local development** (laptop/desktop) and **HPC (Spartan)** environments.

- **GitHub = source of truth** → contains only reproducible workflow elements: source code, configs, scripts, and documentation.
- **Runtime artifacts = environment-specific** → datasets, models, logs, checkpoints, and caches are kept out of version control via `.gitignore`.
- This ensures the repo stays **lightweight, portable, and reproducible**, while heavy or environment-specific files remain local/HPC-only.

---

## 2. Directory Structure (Shared vs. Ignored)

| Directory       | Purpose                                   | Synced (Git)             | Notes                                                          |
| --------------- | ----------------------------------------- | ------------------------ | -------------------------------------------------------------- |
| `configs/`      | YAML config files for models & training   | ✅                       | Templates tracked; env-specific overrides ignored.             |
| `data/`         | Raw + processed datasets                  | 🚫 (ignored)             | Only `README.md` + `.gitkeep` committed. Sync manually to HPC. |
| `docs/`         | Documentation & notes                     | ✅                       | Build artifacts ignored.                                       |
| `environments/` | Setup scripts for HPC/local               | ✅ (scripts) / 🚫 (venv) | Virtual environments ignored.                                  |
| `logs/`         | Training logs & SLURM outputs             | 🚫 (ignored)             | Only `.gitkeep` committed. Keeps repo clean of run artifacts.  |
| `models/`       | Model weights, checkpoints                | 🚫 (ignored)             | Only `.gitkeep` committed. Managed per environment.            |
| `results/`      | Experiment outputs & evaluation           | 🚫 (ignored)             | Sync/backup separately outside Git.                            |
| `scripts/`      | Training & evaluation scripts             | ✅                       | Includes SLURM job scripts for HPC.                            |
| `src/`          | Source code (trainers, utils, evaluation) | ✅                       | Fully tracked.                                                 |
| `temp_storage/` | Temporary HPC runtime cache               | 🚫 (ignored)             | Holds HuggingFace, torch, wandb, temp logs.                    |
| `tests/`        | Unit/integration tests                    | ✅                       | Test artifacts ignored.                                        |

---

## 3. Key `.gitignore` Policies

- **Critical artifacts excluded**

  - Datasets (`data/`)
  - Trained models (`models/`)
  - Training logs (`logs/`)
  - Experimental results (`results/`)
  - HPC temp storage (`temp_storage/`)

- **Environment separation**

  - Python venvs (`venv/`, `environments/venv/`)
  - SLURM job outputs (`*.out`, `*.err`)

- **ML-specific cache exclusion**

  - HuggingFace cache, torch cache, W\&B runs, tensorboard logs, checkpoints, large model weights.

- **Configuration strategy**

  - Track only **templates** (e.g., `*_template.yaml`).
  - Ignore environment-specific overrides (`*_local.yaml`, `*_hpc.yaml`, etc.).

---

## 4. Usage Principles

1. **Clone once per environment** → local machine & HPC each keep their own repo copy.
2. **Push only source of truth** → commit configs, scripts, code, docs.
3. **Keep runtime local** → models, datasets, logs, and caches remain untracked.
4. **Sync datasets/models manually** → via HPC project storage or dedicated dataset/model management.

---

## 5. Why This Design

- Keeps the repo **lightweight and portable**.
- Prevents accidental commits of large files (datasets, checkpoints).
- Separates **workflow definition (GitHub)** from **workflow execution (local/HPC)**.
- Facilitates collaboration without bloating version history.
