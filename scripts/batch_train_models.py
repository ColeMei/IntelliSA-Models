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

# Import ParameterResolver from individual training for consistent parameter resolution
sys.path.append(str(Path(__file__).parent.parent))
try:
    from scripts.train_models import ParameterResolver
except ImportError:
    # Fallback if import fails
    class ParameterResolver:
        def __init__(self, approach: str, config_data: Dict[str, Any]):
            self.approach = approach
            self.config_data = config_data

            # Approach-specific defaults (same as in train_models.py)
            self.approach_defaults = {
                "generative": {
                    "batch_size": 1,
                    "num_epochs": 3,
                    "warmup_steps": 100,
                    "save_steps": 100,
                    "eval_steps": 50,
                    "gradient_accumulation_steps": 4,
                    "weight_decay": 0.01,
                },
                "encoder": {
                    "batch_size": 8,
                    "num_epochs": 3,
                    "warmup_steps": 100,
                    "save_steps": 100,
                    "eval_steps": 50,
                    "weight_decay": 0.01,
                },
            }

        def resolve(self, param_name: str, cli_value: Any, parser_default: Any) -> Any:
            """Resolve parameter value using priority: CLI > YAML > approach default > parser default."""
            # If CLI value differs from parser default, use CLI value
            if cli_value != parser_default:
                return cli_value

            # Try YAML config
            yaml_value = self.config_data.get(param_name)
            if yaml_value is not None:
                return yaml_value

            # Try approach default
            approach_defaults = self.approach_defaults.get(self.approach, {})
            if param_name in approach_defaults:
                return approach_defaults[param_name]

            # Fall back to parser default
            return parser_default

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

        # Base parameters that all experiments have
        base_params = ['learning_rates', 'batch_sizes', 'num_epochs']
        if 'weight_decay' in hyperparams:
            base_params.append('weight_decay')

        # Optional parameters
        optional_params = []
        if 'gradient_accumulation_steps' in hyperparams:
            optional_params.append('gradient_accumulation_steps')

        # Generate all combinations for base parameters
        base_combinations = itertools.product(*[hyperparams[param] for param in base_params])

        # Generate experiments for each base combination
        for base_combo in base_combinations:
            # Create parameter dict with singular keys (convert plural config names to singular)
            param_dict = {}
            for param_name, param_value in zip(base_params, base_combo):
                # Convert plural config names to singular for consistency
                if param_name == 'learning_rates':
                    param_dict['learning_rate'] = param_value
                elif param_name == 'batch_sizes':
                    param_dict['batch_size'] = param_value
                elif param_name == 'num_epochs':
                    param_dict['num_epochs'] = param_value
                elif param_name == 'weight_decay':
                    param_dict['weight_decay'] = param_value
                else:
                    param_dict[param_name] = param_value

            # Handle optional parameters
            if optional_params:
                optional_combinations = itertools.product(*[hyperparams[param] for param in optional_params])
                for optional_combo in optional_combinations:
                    experiment = self._create_experiment_dict(exp_name, exp_config, global_config, param_dict,
                                                            dict(zip(optional_params, optional_combo)))
                    experiments.append(experiment)
            else:
                experiment = self._create_experiment_dict(exp_name, exp_config, global_config, param_dict)
                experiments.append(experiment)

        return experiments

    def _create_experiment_dict(self, exp_name: str, exp_config: Dict, global_config: Dict,
                               param_dict: Dict, optional_dict: Dict = None) -> Dict[str, Any]:
        """Create experiment dictionary with all parameters."""
        experiment = {
            'name': exp_name,
            'model_name': exp_config['model_name'],
            **global_config,  # Global defaults first
            **param_dict      # Swept parameters override defaults
        }

        # Add optional parameters if provided
        if optional_dict:
            experiment.update(optional_dict)

        # Add model-specific settings
        if 'gradient_checkpointing' in exp_config:
            experiment['gradient_checkpointing'] = exp_config['gradient_checkpointing']
        
        # Add global settings that can be overridden by experiment-specific settings
        if 'lr_scheduler_type' in global_config:
            experiment['lr_scheduler_type'] = global_config['lr_scheduler_type']
        if 'warmup_steps' in global_config:
            experiment['warmup_steps'] = global_config['warmup_steps']

        # Add early stopping if configured
        early_stopping = self.config.get('early_stopping', {})
        if early_stopping:
            experiment['early_stopping_patience'] = early_stopping.get('patience', 3)
            experiment['early_stopping_min_delta'] = early_stopping.get('min_delta', 0.001)
            experiment['early_stopping_metric'] = early_stopping.get('metric', 'f1')
            experiment['early_stopping_mode'] = early_stopping.get('mode', 'max')

        return experiment
    
    def create_experiment_config(self, experiment: Dict[str, Any], output_dir: Path) -> Path:
        """Create individual config file for experiment using consistent parameter resolution."""
        # Format experiment name with hyperparameters
        exp_name_parts = [
            experiment['name'],
            f"lr{experiment['learning_rate']}",
            f"bs{experiment['batch_size']}",
            f"ep{experiment['num_epochs']}"
        ]

        # Add optional parameters to name
        if 'weight_decay' in experiment:
            exp_name_parts.append(f"wd{experiment['weight_decay']}")
        if 'gradient_accumulation_steps' in experiment:
            exp_name_parts.append(f"acc{experiment['gradient_accumulation_steps']}")

        exp_name = "_".join(exp_name_parts)

        # Use ParameterResolver for consistent parameter resolution (CLI > YAML > approach defaults > parser defaults)
        param_resolver = ParameterResolver('encoder', experiment)

        # Define parser defaults (matching individual training)
        parser_defaults = {
            'batch_size': 8,
            'num_epochs': 3,
            'warmup_steps': 100,
            'save_steps': 100,
            'eval_steps': 50,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'fp16': True,
            'dataloader_pin_memory': True
        }

        # Resolve parameters using the same logic as individual training
        resolved_batch_size = param_resolver.resolve('batch_size', experiment.get('batch_size', parser_defaults['batch_size']), parser_defaults['batch_size'])
        resolved_num_epochs = param_resolver.resolve('num_epochs', experiment.get('num_epochs', parser_defaults['num_epochs']), parser_defaults['num_epochs'])
        resolved_warmup_steps = param_resolver.resolve('warmup_steps', experiment.get('warmup_steps', parser_defaults['warmup_steps']), parser_defaults['warmup_steps'])
        resolved_save_steps = param_resolver.resolve('save_steps', experiment.get('save_steps', parser_defaults['save_steps']), parser_defaults['save_steps'])
        resolved_eval_steps = param_resolver.resolve('eval_steps', experiment.get('eval_steps', parser_defaults['eval_steps']), parser_defaults['eval_steps'])
        resolved_weight_decay = param_resolver.resolve('weight_decay', experiment.get('weight_decay', parser_defaults['weight_decay']), parser_defaults['weight_decay'])
        resolved_fp16 = param_resolver.resolve('fp16', experiment.get('fp16', parser_defaults['fp16']), parser_defaults['fp16'])
        resolved_pin_memory = param_resolver.resolve('dataloader_pin_memory', experiment.get('dataloader_pin_memory', parser_defaults['dataloader_pin_memory']), parser_defaults['dataloader_pin_memory'])

        config_content = {
            'model_name': experiment['model_name'],
            'max_length': experiment.get('max_length', 256),  # Default to 256 if not specified
            'batch_size': resolved_batch_size,
            'learning_rate': experiment['learning_rate'],  # CLI override has highest priority
            'num_epochs': resolved_num_epochs,
            'weight_decay': resolved_weight_decay,
            'warmup_steps': resolved_warmup_steps,
            'eval_steps': resolved_eval_steps,
            'save_steps': resolved_save_steps,
            'train_path': experiment['train_path'],
            'val_path': experiment['val_path'],
            'output_dir': str(output_dir),
            'evaluation_strategy': experiment.get('evaluation_strategy', 'steps'),
            'save_strategy': experiment.get('save_strategy', 'steps'),
            'logging_steps': experiment.get('logging_steps', 10),
            'load_best_model_at_end': experiment.get('load_best_model_at_end', True),
            'metric_for_best_model': experiment.get('metric_for_best_model', 'f1'),
            'greater_is_better': experiment.get('greater_is_better', True),
            'fp16': resolved_fp16,
            'dataloader_pin_memory': resolved_pin_memory
        }

        # Add optional parameters
        if 'gradient_accumulation_steps' in experiment:
            config_content['gradient_accumulation_steps'] = experiment['gradient_accumulation_steps']
        if 'gradient_checkpointing' in experiment:
            config_content['gradient_checkpointing'] = experiment['gradient_checkpointing']
        if 'early_stopping_patience' in experiment:
            config_content['early_stopping_patience'] = experiment['early_stopping_patience']
            config_content['early_stopping_min_delta'] = experiment['early_stopping_min_delta']
            config_content['early_stopping_metric'] = experiment.get('early_stopping_metric', 'f1')
            config_content['early_stopping_mode'] = experiment.get('early_stopping_mode', 'max')
        if 'lr_scheduler_type' in experiment:
            config_content['lr_scheduler_type'] = experiment['lr_scheduler_type']
        
        # Add threshold sweep config if available
        threshold_sweep = self.config.get('threshold_sweep', {})
        if threshold_sweep:
            config_content['threshold_sweep'] = threshold_sweep

        # Build hyperparameters dict for metadata
        hyperparams = {
            'learning_rate': experiment['learning_rate'],
            'batch_size': resolved_batch_size,
            'num_epochs': resolved_num_epochs
        }
        if 'weight_decay' in experiment:
            hyperparams['weight_decay'] = resolved_weight_decay
        if 'gradient_accumulation_steps' in experiment:
            hyperparams['gradient_accumulation_steps'] = experiment['gradient_accumulation_steps']

        config_content['experiment_metadata'] = {
            'name': exp_name,
            'batch_training_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'original_config': str(self.config_path),
            'model_name': experiment['model_name'],
            'hyperparameters': hyperparams,
            'training_started': datetime.now().isoformat()
        }

        config_path = output_dir / f"{exp_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False, sort_keys=False)

        return config_path
    
    def submit_job(self, config_path: Path, exp_name: str, job_index: int) -> str:
        """Submit a single training job to SLURM."""
        # Use encoder training script (since we're focusing on encoder models)
        slurm_script = f"{self.project_root}/scripts/slurm/batch_encoder_training.slurm"

        # Build SLURM command with hardcoded parameters
        cmd = [
            "sbatch",
            f"--job-name=batch_encoder_{exp_name}_{job_index}",
            "--partition=gpu-a100",
            "--qos=normal",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=32G",
            "--time=0-4:00:00",
            "--tmp=30GB",
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
            filter_pattern: str = None, exclude_pattern: str = None) -> List[str]:
        """Run batch training."""
        experiments = self.generate_experiments()
        
        # Apply filters (support comma-separated patterns)
        if filter_pattern:
            # Split on comma and strip whitespace
            filter_patterns = [p.strip() for p in filter_pattern.split(',')]
            experiments = [exp for exp in experiments
                          if any(pattern in exp['name'] for pattern in filter_patterns)]

        # Apply exclusions (support comma-separated patterns)
        if exclude_pattern:
            # Split on comma and strip whitespace
            exclude_patterns = [p.strip() for p in exclude_pattern.split(',')]
            experiments = [exp for exp in experiments
                          if not any(pattern in exp['name'] for pattern in exclude_patterns)]

        if max_experiments:
            experiments = experiments[:max_experiments]
        
        logger.info(f"Generated {len(experiments)} experiments")
        
        if dry_run:
            logger.info("Dry run - showing experiments:")
            for i, exp in enumerate(experiments):
                exp_name_parts = [
                    exp['name'],
                    f"lr{exp['learning_rate']}",
                    f"bs{exp['batch_size']}",
                    f"ep{exp['num_epochs']}"
                ]

                # Add weight_decay if it varies
                if 'weight_decay' in exp:
                    exp_name_parts.append(f"wd{exp['weight_decay']}")

                # Add gradient_accumulation_steps if it varies
                if 'gradient_accumulation_steps' in exp:
                    exp_name_parts.append(f"acc{exp['gradient_accumulation_steps']}")

                exp_name = "_".join(exp_name_parts)
                logger.info(f"  {i+1}. {exp_name}")
            return []
        
        # Create output directories
        output_base = Path(self.config['output']['base_dir'])
        configs_base = Path(self.config['output'].get('configs_dir', 'configs/generated'))

        output_base.mkdir(parents=True, exist_ok=True)
        configs_base.mkdir(parents=True, exist_ok=True)
        
        # Submit jobs
        job_ids = []
        for i, experiment in enumerate(experiments):
            exp_name_parts = [
                experiment['name'],
                f"lr{experiment['learning_rate']}",
                f"bs{experiment['batch_size']}",
                f"ep{experiment['num_epochs']}"
            ]

            # Add weight_decay if it varies
            if 'weight_decay' in experiment:
                exp_name_parts.append(f"wd{experiment['weight_decay']}")

            # Add gradient_accumulation_steps if it varies
            if 'gradient_accumulation_steps' in experiment:
                exp_name_parts.append(f"acc{experiment['gradient_accumulation_steps']}")

            exp_name = "_".join(exp_name_parts)

            # Create config file in configs directory
            # The config's output_dir will be overridden by SLURM script with timestamped path
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
                       help="Filter experiments by pattern(s) - comma-separated (e.g., 'codebert,codet5_base')")
    parser.add_argument("--exclude", type=str,
                       help="Exclude experiments matching pattern(s) - comma-separated (e.g., 'codet5_large,codet5p_2b')")
    
    args = parser.parse_args()
    
    # Initialize batch trainer
    trainer = BatchTrainer(args.config)
    
    # Run batch training
    job_ids = trainer.run(
        dry_run=args.dry_run,
        max_experiments=args.max_experiments,
        filter_pattern=args.filter,
        exclude_pattern=args.exclude
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
