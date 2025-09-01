#!/usr/bin/env python3
"""
Improved batch training script for hyperparameter sweep.
Generates experiments from config and submits them efficiently.
"""

import argparse
import itertools
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchTrainer:
    def __init__(self, config_path: str):
        """Initialize batch trainer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
        
    def _load_config(self) -> Dict[str, Any]:
        """Load batch training configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate all experiment combinations from config."""
        experiments = []
        global_config = self.config.get('global', {})
        
        # Generate experiments from the config
        exp_configs = self.config.get('experiments', {})
        for exp_name, exp_config in exp_configs.items():
            experiments.extend(self._generate_experiment_combinations(
                exp_name, exp_config, global_config
            ))
        
        return experiments
    
    def _generate_experiment_combinations(self, exp_name: str, exp_config: Dict, 
                                        global_config: Dict) -> List[Dict[str, Any]]:
        """Generate combinations for a specific experiment."""
        experiments = []
        hyperparams = exp_config['hyperparameters']
        
        # Generate all combinations
        combinations = itertools.product(
            hyperparams['learning_rates'],
            hyperparams['batch_sizes'],
            hyperparams['num_epochs']
        )
        
        for lr, bs, epochs in combinations:
            experiment = {
                'name': exp_name,
                'model_name': exp_config['model_name'],
                'learning_rate': lr,
                'batch_size': bs,
                'num_epochs': epochs,
                **global_config
            }
            experiments.append(experiment)
        
        return experiments
    
    def create_experiment_config(self, experiment: Dict[str, Any], output_dir: Path) -> Path:
        """Create individual config file for experiment."""
        # Format experiment name with hyperparameters
        exp_name = f"{experiment['name']}_lr{experiment['learning_rate']}_bs{experiment['batch_size']}_ep{experiment['num_epochs']}"
        
        config_content = {
            'model_name': experiment['model_name'],
            'max_length': experiment['max_length'],
            'batch_size': experiment['batch_size'],
            'learning_rate': experiment['learning_rate'],
            'num_epochs': experiment['num_epochs'],
            'warmup_steps': experiment['warmup_steps'],
            'eval_steps': experiment['eval_steps'],
            'save_steps': experiment['save_steps'],
            'train_path': experiment['train_path'],
            'val_path': experiment['val_path'],
            'output_dir': str(output_dir),
            'evaluation_strategy': experiment['evaluation_strategy'],
            'save_strategy': experiment['save_strategy'],
            'logging_steps': experiment['logging_steps'],
            'load_best_model_at_end': experiment['load_best_model_at_end'],
            'metric_for_best_model': experiment['metric_for_best_model'],
            'greater_is_better': experiment['greater_is_better'],
            'experiment_metadata': {
                'name': exp_name,
                'batch_training_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'original_config': str(self.config_path),
                'model_name': experiment['model_name'],
                'hyperparameters': {
                    'learning_rate': experiment['learning_rate'],
                    'batch_size': experiment['batch_size'],
                    'num_epochs': experiment['num_epochs']
                },
                'training_started': datetime.now().isoformat()
            }
        }
        
        config_path = output_dir / f"{exp_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False, sort_keys=False)
        
        return config_path
    
    def submit_job(self, config_path: Path, exp_name: str, job_index: int) -> str:
        """Submit a single training job to SLURM."""
        slurm_config = self.config.get('slurm', {})
        
        # Use encoder training script (since we're focusing on encoder models)
        slurm_script = f"{self.project_root}/scripts/slurm/batch_encoder_training.slurm"
        
        # Build SLURM command
        cmd = [
            "sbatch",
            f"--job-name=batch_encoder_{exp_name}_{job_index}",
            f"--partition={slurm_config.get('partition', 'gpu-a100')}",
            f"--qos={slurm_config.get('qos', 'normal')}",
            f"--gres={slurm_config.get('gres', 'gpu:1')}",
            f"--cpus-per-task={slurm_config.get('cpus_per_task', 4)}",
            f"--mem={slurm_config.get('mem', '32G')}",
            f"--time={slurm_config.get('time', '0-4:00:00')}",
            f"--tmp={slurm_config.get('tmp', '30GB')}",
            f"--output={self.project_root}/logs/slurm_outputs/batch_encoder_{exp_name}_{job_index}_%j.out",
            f"--error={self.project_root}/logs/slurm_outputs/batch_encoder_{exp_name}_{job_index}_%j.err",
            "--mail-type=BEGIN,END,FAIL",
            "--mail-user=qmmei@student.unimelb.edu.au",
            "--export=ALL",
            slurm_script,
            str(config_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"✅ Submitted {exp_name}: Job ID {job_id}")
            return job_id
        else:
            logger.error(f"❌ Failed to submit {exp_name}: {result.stderr}")
            return None
    
    def run(self, dry_run: bool = False, max_experiments: int = None, 
            filter_pattern: str = None) -> List[str]:
        """Run batch training."""
        experiments = self.generate_experiments()
        
        # Apply filters
        if filter_pattern:
            experiments = [exp for exp in experiments if filter_pattern in exp['name']]
        
        if max_experiments:
            experiments = experiments[:max_experiments]
        
        logger.info(f"Generated {len(experiments)} experiments")
        
        if dry_run:
            logger.info("Dry run - showing experiments:")
            for i, exp in enumerate(experiments):
                exp_name = f"{exp['name']}_lr{exp['learning_rate']}_bs{exp['batch_size']}_ep{exp['num_epochs']}"
                logger.info(f"  {i+1}. {exp_name}")
            return []
        
        # Create output directories
        output_base = Path(self.config['output']['base_dir'])
        configs_base = Path(self.config['output']['configs_dir'])
        
        output_base.mkdir(parents=True, exist_ok=True)
        configs_base.mkdir(parents=True, exist_ok=True)
        
        # Submit jobs
        job_ids = []
        for i, experiment in enumerate(experiments):
            exp_name = f"{experiment['name']}_lr{experiment['learning_rate']}_bs{experiment['batch_size']}_ep{experiment['num_epochs']}"
            
            # Create experiment-specific directory
            exp_output_dir = output_base / exp_name
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create config file
            config_path = self.create_experiment_config(experiment, configs_base)
            
            # Submit job
            job_id = self.submit_job(config_path, exp_name, i)
            if job_id:
                job_ids.append(job_id)
        
        return job_ids

def main():
    parser = argparse.ArgumentParser(description="Batch training for hyperparameter sweep")
    parser.add_argument("--config", default="configs/encoder/batch_training_config.yaml", 
                       help="Batch training config file")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show experiments without running")
    parser.add_argument("--max-experiments", type=int, 
                       help="Limit number of experiments")
    parser.add_argument("--filter", type=str, 
                       help="Filter experiments by pattern (e.g., 'codebert')")
    
    args = parser.parse_args()
    
    # Initialize batch trainer
    trainer = BatchTrainer(args.config)
    
    # Run batch training
    job_ids = trainer.run(
        dry_run=args.dry_run,
        max_experiments=args.max_experiments,
        filter_pattern=args.filter
    )
    
    if job_ids:
        logger.info(f"🚀 Submitted {len(job_ids)} jobs: {job_ids}")
        
        # Save job IDs for tracking
        job_file = Path("logs/batch_training_jobs.txt")
        job_file.parent.mkdir(parents=True, exist_ok=True)
        with open(job_file, 'w') as f:
            f.write(f"Batch training started at {datetime.now()}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Jobs: {', '.join(job_ids)}\n")
        
        logger.info(f"Job IDs saved to {job_file}")

if __name__ == "__main__":
    main()
