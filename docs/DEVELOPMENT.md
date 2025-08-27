# Development Guide

## Local Development Setup

1. Run the setup script:
   ```bash
   bash setup_local.sh
   ```

2. Set up Python environment:
   ```bash
   bash environments/setup_venv.sh
   source environments/venv/bin/activate
   ```

3. Run tests:
   ```bash
   python -m pytest tests/
   ```

## Project Structure

- `src/`: Main source code
- `scripts/`: Training and utility scripts
- `configs/`: Model and training configurations
- `data/`: Dataset storage (not synced to git)
- `models/`: Trained model outputs (not synced to git)
- `logs/`: Training logs (not synced to git)

## Development Workflow

1. Develop and test locally
2. Commit changes to git
3. Deploy to HPC via `git pull`
4. Run training on HPC
5. Sync results back if needed
