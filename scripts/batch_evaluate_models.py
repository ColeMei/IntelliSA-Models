#!/usr/bin/env python3
"""
Batch evaluation of all trained models from batch training using SLURM.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchEvaluator:
    def __init__(self, models_dir: str = "models/experiments/encoder",
                 batch_config_path: str = "configs/encoder/batch_evaluation_config.yaml"):
        """Initialize batch evaluator."""
        self.models_dir = Path(models_dir)
        self.project_root = Path(__file__).parent.parent
        self.batch_config_path = Path(batch_config_path)
        self.batch_config = self._load_batch_config()

    def _load_batch_config(self) -> Dict[str, Any]:
        """Load batch evaluation configuration."""
        if self.batch_config_path.exists():
            with open(self.batch_config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Batch config {self.batch_config_path} not found, using defaults")
            return {}

    def find_trained_models(self) -> List[Dict[str, Any]]:
        """Find all trained model directories."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return []

        trained_models = []

        # Look for model directories
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Skip symlinks to avoid duplicate evaluations
            if model_dir.is_symlink():
                continue
                
            # Check if this is a trained model (has config and model files)
            config_file = model_dir / "config_used.yaml"
            if not config_file.exists():
                continue
                
            # Check if model files exist
            model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
            if not model_files:
                continue
                
            try:
                # Load config to get experiment info
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                metadata = config.get('experiment_metadata', {})
                exp_name = metadata.get('name', model_dir.name)
                
                trained_models.append({
                    'name': exp_name,
                    'path': str(model_dir),
                    'config_file': str(config_file),
                    'model_name': config.get('model_name', 'unknown')
                })
                
            except Exception as e:
                logger.warning(f"Failed to load config from {model_dir}: {e}")
        
        return sorted(trained_models, key=lambda x: x['name'])

    def _create_custom_evaluation_config(self, model_info: Dict[str, Any], eval_dir: Path) -> Path:
        """Create a custom evaluation config for a specific model."""
        # Load base evaluation config
        base_config_path = self.project_root / "configs" / "evaluation_config.yaml"
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update encoder model path
        if 'encoder' in config.get('models', {}):
            config['models']['encoder']['path'] = model_info['path']

            # Use standard batch size logic (simplified)
            model_name = model_info['model_name'].lower()
            if 'large' in model_name or '2b' in model_name:
                config['models']['encoder']['batch_size'] = 4
            else:
                config['models']['encoder']['batch_size'] = 8

        # Update output directory
        config['output_dir'] = str(eval_dir)

        # Save custom config
        custom_config_path = eval_dir / f"eval_config_{model_info['name']}.yaml"
        with open(custom_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return custom_config_path



    def submit_evaluation_job(self, model_info: Dict[str, Any], output_dir: Path, job_index: int) -> str:
        """Submit evaluation job to SLURM."""
        logger.info(f"Submitting evaluation job for {model_info['name']}...")

        # Create evaluation output directory
        eval_dir = output_dir / model_info['name']
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Create custom evaluation config for this specific model
        custom_config = self._create_custom_evaluation_config(model_info, eval_dir)

        # Submit SLURM job using the existing evaluation script
        cmd = [
            "sbatch",
            f"--job-name=batch_eval_{model_info['name']}_{job_index}",
            "--partition=gpu-a100",
            "--qos=normal",
            "--gres=gpu:1",
            "--cpus-per-task=8",
            "--mem=32G",
            "--time=0-2:00:00",
            "--tmp=10GB",
            f"--output={self.project_root}/logs/slurm_outputs/batch_eval_{model_info['name']}_{job_index}_%j.out",
            f"--error={self.project_root}/logs/slurm_outputs/batch_eval_{model_info['name']}_{job_index}_%j.err",
            "--mail-type=BEGIN,END,FAIL",
            "--mail-user=qmmei@student.unimelb.edu.au",
            "--export=ALL",
            f"{self.project_root}/scripts/slurm/evaluate_models.slurm",
            str(custom_config),
            "encoder"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"✅ Submitted evaluation for {model_info['name']}: Job ID {job_id}")
            return job_id
        else:
            logger.error(f"❌ Failed to submit evaluation for {model_info['name']}: {result.stderr}")
            return None
    
    def run_batch_evaluation(self, output_dir: str = "results/experiments/evaluation",
                           max_models: int = None, filter_pattern: str = None) -> List[str]:
        """Run batch evaluation of all models using SLURM."""
        models = self.find_trained_models()
        
        if not models:
            logger.error("No trained models found to evaluate")
            return []
        
        # Apply filters
        if filter_pattern:
            models = [m for m in models if filter_pattern in m['name']]
        
        if max_models:
            models = models[:max_models]
        
        logger.info(f"Found {len(models)} models to evaluate")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Submit evaluation jobs with spacing
        job_ids = []
        job_spacing = 1  # Default 1 second spacing between job submissions

        for i, model_info in enumerate(models):
            logger.info(f"Submitting evaluation job {i+1}/{len(models)}: {model_info['name']}")

            job_id = self.submit_evaluation_job(model_info, output_path, i)
            if job_id:
                job_ids.append(job_id)

            # Add spacing between job submissions to avoid SLURM overload
            if i < len(models) - 1 and job_spacing > 0:
                import time
                logger.info(f"Waiting {job_spacing}s before next job submission...")
                time.sleep(job_spacing)

        logger.info(f"Batch evaluation submitted: {len(job_ids)}/{len(models)} jobs")
        return job_ids

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of trained models using SLURM")
    parser.add_argument("--models-dir", default="models/experiments/encoder",
                       help="Directory containing trained models")
    parser.add_argument("--output-dir", default="results/experiments/evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--batch-config", default="configs/encoder/batch_evaluation_config.yaml",
                       help="Batch evaluation configuration file")

    parser.add_argument("--max-models", type=int,
                       help="Maximum number of models to evaluate")
    parser.add_argument("--filter", type=str,
                       help="Filter models by pattern (e.g., 'codebert')")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show models to evaluate without running")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BatchEvaluator(args.models_dir, args.batch_config)
    
    # Find models
    models = evaluator.find_trained_models()
    
    if not models:
        logger.error("No trained models found")
        return
    
    # Apply filters
    if args.filter:
        models = [m for m in models if args.filter in m['name']]
    
    if args.max_models:
        models = models[:args.max_models]
    
    logger.info(f"Found {len(models)} models to evaluate")
    
    if args.dry_run:
        logger.info("Models to evaluate:")
        for i, model in enumerate(models):
            logger.info(f"  {i+1}. {model['name']} ({model['model_name']})")
        return
    
    # Run batch evaluation
    job_ids = evaluator.run_batch_evaluation(
        output_dir=args.output_dir,
        max_models=args.max_models,
        filter_pattern=args.filter
    )
    
    # Save job IDs for tracking
    if job_ids:
        logger.info(f"Batch evaluation submitted successfully!")
        logger.info(f"Submitted {len(job_ids)} evaluation jobs: {job_ids}")
        
        job_file = Path("logs/batch_evaluation_jobs.txt")
        job_file.parent.mkdir(parents=True, exist_ok=True)
        with open(job_file, 'w') as f:
            f.write(f"Batch evaluation started at {datetime.now()}\n")
            f.write(f"Models directory: {args.models_dir}\n")
            f.write(f"Jobs: {', '.join(job_ids)}\n")
        
        logger.info(f"Job IDs saved to {job_file}")

if __name__ == "__main__":
    main()
