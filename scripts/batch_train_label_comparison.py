#!/usr/bin/env python3
"""
Batch training script for label comparison experiment.
Trains champion model configuration on multiple label variants.
"""

import argparse
import itertools
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Ensure repo src is on path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_defaults import apply_encoder_defaults
from src.utils.parameter_resolver import ParameterResolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelComparisonTrainer:
    def __init__(self, config_path: str, dataset_variants: List[str]):
        """Initialize label comparison trainer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.dataset_variants = dataset_variants
        self.project_root = Path(__file__).parent.parent
        
    def _load_config(self) -> Dict[str, Any]:
        """Load champion configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return apply_encoder_defaults(config)
    
    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate all experiment combinations."""
        experiments = []
        global_config = self.config.get('global', {})
        
        # Get the experiment config (should only be one for champion)
        exp_configs = self.config.get('experiments', {})
        if len(exp_configs) != 1:
            logger.warning(f"Expected 1 experiment config, found {len(exp_configs)}. Using first one.")
        
        exp_name, exp_config = next(iter(exp_configs.items()))
        
        # Generate experiments for each dataset variant
        for dataset_variant in self.dataset_variants:
            experiments.extend(
                self._generate_experiment_combinations(
                    exp_name, exp_config, global_config, dataset_variant
                )
            )
        
        return experiments
    
    def _generate_experiment_combinations(
        self, exp_name: str, exp_config: Dict, 
        global_config: Dict, dataset_variant: str
    ) -> List[Dict[str, Any]]:
        """Generate combinations for a specific experiment and dataset variant."""
        experiments = []
        hyperparams = exp_config['hyperparameters']
        
        # Get hyperparameter values (should be single values for champion)
        learning_rate = hyperparams['learning_rates'][0]
        batch_size = hyperparams['batch_sizes'][0]
        num_epochs = hyperparams['num_epochs'][0]
        weight_decay = hyperparams['weight_decay'][0]
        grad_accum = hyperparams.get('gradient_accumulation_steps', [1])[0]
        seeds = hyperparams.get('seeds', [41, 42, 43])
        
        # Generate one experiment per seed
        for seed in seeds:
            experiment = {
                'name': f"{exp_name}_{dataset_variant}_seed{seed}",
                'dataset_variant': dataset_variant,
                'model_name': exp_config['model_name'],
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'weight_decay': weight_decay,
                'gradient_accumulation_steps': grad_accum,
                'seed': seed,
                'lr_scheduler_type': global_config.get('lr_scheduler_type', 'cosine'),
                'warmup_steps': global_config.get('warmup_steps', 500),
            }
            
            # Add early stopping config if present
            early_stopping = self.config.get('early_stopping', {})
            if early_stopping.get('enabled', False):
                experiment['early_stopping_patience'] = early_stopping.get('patience', 2)
                experiment['early_stopping_min_delta'] = early_stopping.get('min_delta', 0.001)
                experiment['early_stopping_metric'] = early_stopping.get('metric', 'f1')
                experiment['early_stopping_mode'] = early_stopping.get('mode', 'max')
            
            # Add threshold sweep config if present
            threshold_sweep = self.config.get('threshold_sweep', {})
            if threshold_sweep.get('enabled', False):
                experiment['threshold_sweep'] = threshold_sweep
            
            experiments.append(experiment)
        
        return experiments
    
    def create_experiment_config(self, experiment: Dict[str, Any]) -> Path:
        """Create individual config file for experiment."""
        exp_name = experiment['name']
        output_dir = self.project_root / "configs" / "label_comparison" / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ParameterResolver for consistent parameter resolution
        param_resolver = ParameterResolver('encoder', experiment)
        
        # Define parser defaults (matching individual training)
        parser_defaults = {
            'batch_size': 8,
            'num_epochs': 3,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'eval_steps': 100,
            'save_steps': 100,
            'fp16': True,
            'dataloader_pin_memory': True
        }
        
        # Resolve parameters
        resolved_batch_size = param_resolver.resolve(
            'batch_size',
            experiment.get('batch_size'),
            parser_defaults['batch_size']
        )
        resolved_num_epochs = param_resolver.resolve(
            'num_epochs',
            experiment.get('num_epochs'),
            parser_defaults['num_epochs']
        )
        resolved_weight_decay = param_resolver.resolve(
            'weight_decay',
            experiment.get('weight_decay'),
            parser_defaults['weight_decay']
        )
        resolved_warmup_steps = param_resolver.resolve(
            'warmup_steps',
            experiment.get('warmup_steps'),
            parser_defaults['warmup_steps']
        )
        resolved_eval_steps = param_resolver.resolve(
            'eval_steps',
            experiment.get('eval_steps', 100),
            parser_defaults['eval_steps']
        )
        resolved_save_steps = param_resolver.resolve(
            'save_steps',
            experiment.get('save_steps', 100),
            parser_defaults['save_steps']
        )
        resolved_fp16 = param_resolver.resolve(
            'fp16',
            experiment.get('fp16', True),
            parser_defaults['fp16']
        )
        resolved_pin_memory = param_resolver.resolve(
            'dataloader_pin_memory',
            experiment.get('dataloader_pin_memory', True),
            parser_defaults['dataloader_pin_memory']
        )
        
        # Build config content
        config_content = {
            'model_name': experiment['model_name'],
            'learning_rate': experiment['learning_rate'],
            'batch_size': resolved_batch_size,
            'num_epochs': resolved_num_epochs,
            'weight_decay': resolved_weight_decay,
            'gradient_accumulation_steps': experiment['gradient_accumulation_steps'],
            'lr_scheduler_type': experiment['lr_scheduler_type'],
            'warmup_steps': resolved_warmup_steps,
            'eval_steps': resolved_eval_steps,
            'save_steps': resolved_save_steps,
            'seed': experiment['seed'],
            'train_path': f"{self.project_root}/data/{experiment['dataset_variant']}/train.jsonl",
            'val_path': f"{self.project_root}/data/{experiment['dataset_variant']}/val.jsonl",
            'output_dir': str(output_dir / "model"),
            'evaluation_strategy': 'steps',
            'save_strategy': 'steps',
            'logging_steps': 10,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'f1',
            'greater_is_better': True,
            'fp16': resolved_fp16,
            'dataloader_pin_memory': resolved_pin_memory
        }
        
        # Add early stopping if configured
        if 'early_stopping_patience' in experiment:
            config_content['early_stopping_patience'] = experiment['early_stopping_patience']
            config_content['early_stopping_min_delta'] = experiment['early_stopping_min_delta']
            config_content['early_stopping_metric'] = experiment['early_stopping_metric']
            config_content['early_stopping_mode'] = experiment['early_stopping_mode']
        
        # Add threshold sweep if configured
        if 'threshold_sweep' in experiment:
            config_content['threshold_sweep'] = experiment['threshold_sweep']
        
        # Add metadata
        config_content['experiment_metadata'] = {
            'name': exp_name,
            'experiment_type': 'label_comparison',
            'dataset_variant': experiment['dataset_variant'],
            'batch_training_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'original_config': str(self.config_path),
            'model_name': experiment['model_name'],
            'hyperparameters': {
                'learning_rate': experiment['learning_rate'],
                'batch_size': resolved_batch_size,
                'num_epochs': resolved_num_epochs,
                'weight_decay': resolved_weight_decay,
                'gradient_accumulation_steps': experiment['gradient_accumulation_steps'],
                'seed': experiment['seed']
            },
            'training_started': datetime.now().isoformat()
        }
        
        config_path = output_dir / f"{exp_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False, sort_keys=False)
        
        return config_path
    
    def submit_job(self, config_path: Path, experiment: Dict[str, Any], job_index: int) -> str:
        """Submit a single training job to SLURM."""
        exp_name = experiment['name']
        dataset_variant = experiment['dataset_variant']
        slurm_script = f"{self.project_root}/scripts/slurm/label_comparison_training.slurm"
        
        # Build SLURM command
        cmd = [
            "sbatch",
            f"--job-name=label_comp_{exp_name}_{job_index}",
            "--partition=gpu-a100",
            "--qos=normal",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=32G",
            "--time=0-4:00:00",
            "--tmp=30GB",
            f"--output={self.project_root}/logs/slurm_outputs/label_comp_{exp_name}_{job_index}_%j.out",
            f"--error={self.project_root}/logs/slurm_outputs/label_comp_{exp_name}_{job_index}_%j.err",
            "--mail-type=BEGIN,END,FAIL",
            "--mail-user=qmmei@student.unimelb.edu.au",
            "--export=ALL",
            slurm_script,
            str(config_path),
            dataset_variant
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"✅ Submitted {exp_name} ({dataset_variant}): Job ID {job_id}")
            return job_id
        else:
            logger.error(f"❌ Failed to submit {exp_name}: {result.stderr}")
            return None
    
    def run(self, dry_run: bool = False, max_jobs: int = None) -> List[str]:
        """Generate configs and submit all jobs."""
        experiments = self.generate_experiments()
        
        logger.info(f"Generated {len(experiments)} experiments across {len(self.dataset_variants)} label variants")
        
        if dry_run:
            logger.info("\n🔍 DRY RUN - Would submit the following experiments:")
            for i, exp in enumerate(experiments, 1):
                logger.info(f"  {i}. {exp['name']} ({exp['dataset_variant']}) - Seed {exp['seed']}")
            return []
        
        # Limit number of jobs if specified
        if max_jobs:
            experiments = experiments[:max_jobs]
            logger.info(f"⚠️ Limiting to first {max_jobs} experiments")
        
        # Create configs and submit jobs
        job_ids = []
        for i, experiment in enumerate(experiments, 1):
            logger.info(f"\n📝 Creating config {i}/{len(experiments)}: {experiment['name']}")
            config_path = self.create_experiment_config(experiment)
            logger.info(f"   Config saved to: {config_path}")
            
            logger.info(f"🚀 Submitting job {i}/{len(experiments)}...")
            job_id = self.submit_job(config_path, experiment, i)
            if job_id:
                job_ids.append(job_id)
        
        logger.info(f"\n✅ Submitted {len(job_ids)}/{len(experiments)} jobs successfully")
        if job_ids:
            logger.info(f"📊 Job IDs: {', '.join(job_ids)}")
        
        return job_ids


def main():
    parser = argparse.ArgumentParser(description="Batch training for label comparison experiment")
    parser.add_argument(
        "--config", 
        default="configs/encoder/label_comparison_champion.yaml",
        help="Path to champion configuration file"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["processed_label1", "processed_label2", "processed_label3"],
        help="Dataset variants to train on (e.g., processed_label1 processed_label2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs but don't submit jobs"
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Maximum number of jobs to submit"
    )
    
    args = parser.parse_args()
    
    # Verify config exists
    if not Path(args.config).exists():
        logger.error(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Create trainer and run
    trainer = LabelComparisonTrainer(args.config, args.datasets)
    job_ids = trainer.run(dry_run=args.dry_run, max_jobs=args.max_jobs)
    
    if not args.dry_run:
        logger.info(f"\n{'='*60}")
        logger.info("🎯 Label comparison training jobs submitted!")
        logger.info(f"   Total jobs: {len(job_ids)}")
        logger.info(f"   Datasets: {', '.join(args.datasets)}")
        logger.info(f"   Config: {args.config}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
