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

# Import EvaluationParameterResolver from individual evaluation for consistency
sys.path.append(str(Path(__file__).parent.parent))
try:
    from scripts.evaluate_models import EvaluationParameterResolver
except ImportError:
    # Fallback if import fails
    class EvaluationParameterResolver:
        def __init__(self, config_data: Dict[str, Any]):
            self.config_data = config_data

        def resolve_model_params(self, approach: str, args) -> Dict[str, Any]:
            """Resolve model-specific parameters."""
            model_config = self.config_data.get('models', {}).get(approach, {})
            eval_config = self.config_data.get('evaluation', {})

            # Batch size: model-specific > global eval > default
            batch_size = (model_config.get('batch_size') or
                         eval_config.get('batch_size') or
                         8)

            # Max samples: config > None
            max_samples = eval_config.get('max_samples')

            # Save predictions: config > True
            save_predictions = eval_config.get('save_predictions', True)

            params = {
                'batch_size': batch_size,
                'max_samples': max_samples,
                'save_predictions': save_predictions
            }

            # Threshold block (optional)
            thr_cfg = self.config_data.get('evaluation', {}).get('threshold')
            if isinstance(thr_cfg, dict):
                params['threshold'] = thr_cfg
            return params

        def resolve_paths(self, args) -> str:
            """Resolve test path."""
            eval_config = self.config_data.get('evaluation', {})
            return eval_config.get('test_path', 'data/processed/test.jsonl')

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
        # Initialize parameter resolver for consistent parameter handling
        self.param_resolver = EvaluationParameterResolver(self.batch_config)

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
        """Create a custom evaluation config for a specific model using parameter resolver."""
        # Start with batch config as base
        config = self.batch_config.copy()

        # Update encoder model path
        if 'models' in config and 'encoder' in config['models']:
            config['models']['encoder']['path'] = model_info['path']

            # Use parameter resolver to determine optimal batch size
            model_name = model_info['model_name'].lower()
            if 'large' in model_name or '2b' in model_name or '770m' in model_name:
                # Larger models need smaller batch sizes
                config['models']['encoder']['batch_size'] = 4
            else:
                # Smaller models can use larger batch sizes
                config['models']['encoder']['batch_size'] = 8

        # Update output directory
        if 'evaluation' in config:
            config['evaluation']['output_dir'] = str(eval_dir)

        # Save custom config to a shared directory on the project filesystem
        shared_dir = self.project_root / "logs" / "batch_eval_configs"
        shared_dir.mkdir(parents=True, exist_ok=True)
        custom_config_path = shared_dir / f"eval_config_{model_info['name']}.yaml"
        with open(custom_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return custom_config_path



    def submit_evaluation_job(self, model_info: Dict[str, Any], output_dir: Path, job_index: int) -> List[str]:
        """Submit evaluation jobs for all test sets (combined + technology-specific)."""
        logger.info(f"Submitting evaluation jobs for {model_info['name']} on all test sets...")

        job_ids = []
        test_sets = self.batch_config.get('evaluation', {}).get('test_sets', {})

        # Submit job for combined test set
        combined_test_path = test_sets.get('combined')
        if combined_test_path:
            job_id = self._submit_single_evaluation(
                model_info, output_dir, job_index, combined_test_path, "combined"
            )
            if job_id:
                job_ids.append(job_id)

        # Submit jobs for technology-specific test sets
        technologies = test_sets.get('technologies', {})
        for tech_name, tech_test_path in technologies.items():
            job_id = self._submit_single_evaluation(
                model_info, output_dir, job_index, tech_test_path, tech_name
            )
            if job_id:
                job_ids.append(job_id)

        logger.info(f"✅ Submitted {len(job_ids)} evaluation jobs for {model_info['name']}")
        return job_ids

    def _submit_single_evaluation(self, model_info: Dict[str, Any], output_dir: Path,
                                job_index: int, test_path: str, test_set_name: str) -> str:
        """Submit a single evaluation job for a specific test set."""
        # Create evaluation output directory for this test set
        eval_dir = output_dir / model_info['name'] / test_set_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Determine batch size based on model size
        model_name = model_info['model_name'].lower()
        if any(keyword in model_name for keyword in ['large', '2b', '770m']):
            batch_size = 4  # Smaller batch for large models
        else:
            batch_size = 8  # Larger batch for smaller models

        # Threshold env from config if provided
        thr = self.batch_config.get('evaluation', {}).get('threshold', {})
        thr_mode = thr.get('mode') if isinstance(thr, dict) else None
        thr_file = thr.get('file') if isinstance(thr, dict) else None
        thr_fixed = thr.get('fixed') if isinstance(thr, dict) else None
        # Use single threshold for all test sets (no per-technology mapping)
        thr_key = None

        # If mode=file and no file specified, default to per-run sweep file
        if thr_mode == 'file' and (thr_file is None or str(thr_file).strip() == ''):
            thr_file = str(Path(model_info['path']) / 'threshold_sweep_results.json')

        # Hard-check threshold file exists when mode=file to lock Stage 3/4
        if thr_mode == 'file':
            if not thr_file or not Path(thr_file).exists():
                logger.error(f"Threshold mode 'file' requires an existing file. Missing: {thr_file}")
                return None

        # Submit SLURM job with parameters
        cmd = [
            "sbatch",
            f"--job-name=batch_eval_{model_info['name']}_{test_set_name}_{job_index}",
            "--partition=gpu-a100",
            "--qos=normal",
            "--gres=gpu:1",
            "--cpus-per-task=8",
            "--mem=32G",
            "--time=0-2:00:00",
            "--tmp=10GB",
            f"--output={self.project_root}/logs/slurm_outputs/batch_eval_{model_info['name']}_{test_set_name}_{job_index}_%j.out",
            f"--error={self.project_root}/logs/slurm_outputs/batch_eval_{model_info['name']}_{test_set_name}_{job_index}_%j.err",
            "--mail-type=BEGIN,END,FAIL",
            "--mail-user=qmmei@student.unimelb.edu.au",
            "--export=ALL",
            f"{self.project_root}/scripts/slurm/batch_evaluate_models.slurm",
            str(model_info['path']),  # Model path
            str(eval_dir),           # Output directory
            str(batch_size),         # Batch size
            str(test_path),          # Test set path
            str(thr_mode or ''),     # threshold mode
            str(thr_file or ''),     # threshold file path
            str(thr_fixed or ''),    # fixed threshold value
            str(thr_key or '')       # per-dataset key
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"✅ Submitted {model_info['name']} on {test_set_name}: Job ID {job_id}")
            return job_id
        else:
            logger.error(f"❌ Failed to submit {model_info['name']} on {test_set_name}: {result.stderr}")
            return None
    
    def run_batch_evaluation(self, output_dir: str = "results/experiments/evaluation",
                           max_models: int = None, filter_pattern: str = None, exclude_pattern: str = None) -> List[str]:
        """Run batch evaluation of all models using SLURM on all test sets."""
        models = self.find_trained_models()

        if not models:
            logger.error("No trained models found to evaluate")
            return []

        # Apply filters (support comma-separated patterns)
        if filter_pattern:
            # Split on comma and strip whitespace
            filter_patterns = [p.strip() for p in filter_pattern.split(',')]
            models = [m for m in models if any(pattern in m['name'] for pattern in filter_patterns)]

        # Apply exclusions (support comma-separated patterns)
        if exclude_pattern:
            # Split on comma and strip whitespace
            exclude_patterns = [p.strip() for p in exclude_pattern.split(',')]
            models = [m for m in models if not any(pattern in m['name'] for pattern in exclude_patterns)]

        if max_models:
            models = models[:max_models]

        logger.info(f"Found {len(models)} models to evaluate")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate total jobs: each model × each test set
        test_sets = self.batch_config.get('evaluation', {}).get('test_sets', {})
        num_test_sets = 1 + len(test_sets.get('technologies', {}))  # combined + technologies
        total_jobs = len(models) * num_test_sets

        logger.info(f"Will submit {total_jobs} evaluation jobs ({len(models)} models × {num_test_sets} test sets)")

        # Submit evaluation jobs with spacing
        all_job_ids = []
        job_spacing = self.batch_config.get('batch_evaluation', {}).get('job_spacing_seconds', 1)

        for i, model_info in enumerate(models):
            logger.info(f"Submitting evaluation jobs {i+1}/{len(models)}: {model_info['name']}")

            job_ids = self.submit_evaluation_job(model_info, output_path, i)
            all_job_ids.extend(job_ids)

            # Add spacing between job submissions to avoid SLURM overload
            if i < len(models) - 1 and job_spacing > 0:
                import time
                logger.info(f"Waiting {job_spacing}s before next model...")
                time.sleep(job_spacing)

        logger.info(f"Batch evaluation submitted: {len(all_job_ids)}/{total_jobs} jobs")

        # Create technology comparison summary
        self._create_technology_comparison_summary(output_path, models)

        return all_job_ids

    def _create_technology_comparison_summary(self, output_dir: Path, models: List[Dict[str, Any]]):
        """Create a summary comparing model performance across technologies."""
        summary_file = output_dir / "technology_comparison_summary.json"
        readme_file = output_dir / "README.md"

        # Prepare summary structure
        summary = {
            "evaluation_summary": {
                "total_models": len(models),
                "test_sets": ["combined", "chef", "ansible", "puppet"],
                "total_evaluations": len(models) * 4,  # 4 test sets per model
                "generated_at": datetime.now().isoformat()
            },
            "models": [],
            "expected_output_structure": {
                "model_name": {
                    "combined": "path/to/combined/results",
                    "chef": "path/to/chef/results",
                    "ansible": "path/to/ansible/results",
                    "puppet": "path/to/puppet/results"
                }
            }
        }

        # Add model information
        for model in models:
            summary["models"].append({
                "name": model["name"],
                "model_name": model["model_name"],
                "path": model["path"],
                "expected_results": {
                    "combined": f"{model['name']}/combined/encoder_eval/evaluation_results.json",
                    "chef": f"{model['name']}/chef/encoder_eval/evaluation_results.json",
                    "ansible": f"{model['name']}/ansible/encoder_eval/evaluation_results.json",
                    "puppet": f"{model['name']}/puppet/encoder_eval/evaluation_results.json"
                }
            })

        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create README with instructions
        readme_content = f"""# Batch Evaluation Results - Technology Comparison

This directory contains evaluation results for {len(models)} encoder models across 4 test sets (combined + 3 technologies).

## 📊 Overview
- **Models Evaluated**: {len(models)}
- **Test Sets**: combined, chef, ansible, puppet
- **Total Evaluations**: {len(models) * 4}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Directory Structure
```
[model_name]/
├── combined/
│   └── encoder_eval/
│       ├── evaluation_results.json
│       └── detailed_predictions.json
├── chef/
│   └── encoder_eval/
│       ├── evaluation_results.json
│       └── detailed_predictions.json
├── ansible/
│   └── encoder_eval/
│       ├── evaluation_results.json
│       └── detailed_predictions.json
└── puppet/
    └── encoder_eval/
        ├── evaluation_results.json
        └── detailed_predictions.json
```

## 🔍 Results Files
Each evaluation generates:
- `evaluation_results.json`: Comprehensive metrics and statistics
- `detailed_predictions.json`: Individual prediction details
- `evaluation_config.yaml`: Configuration used for evaluation

## 📈 Key Metrics
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## 🤖 Models Evaluated
"""

        for model in models:
            readme_content += f"- **{model['name']}**: {model['model_name']}\n"

        readme_content += """
## 🚀 Next Steps
1. Review individual model performance on each technology
2. Compare cross-technology generalization ability
3. Identify best models for each technology
4. Analyze per-smell performance metrics

## 📋 Usage Examples
```bash
# View results for a specific model and technology
cat codebert_base_lr2e-5_bs16_ep3_wd0.01/chef/encoder_eval/evaluation_results.json

# Compare F1 scores across technologies for one model
python -c "
import json
results = {}
for tech in ['combined', 'chef', 'ansible', 'puppet']:
    with open(f'model_name/{tech}/encoder_eval/evaluation_results.json') as f:
        data = json.load(f)
        results[tech] = data['metrics']['f1']
print('F1 Scores:', results)
"
```

---
*Generated by batch evaluation system*
"""

        with open(readme_file, 'w') as f:
            f.write(readme_content)

        logger.info(f"Technology comparison summary created: {summary_file}")
        logger.info(f"README documentation created: {readme_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of trained models using SLURM")
    # Configuration
    parser.add_argument("--config", default="configs/encoder/batch_evaluation_config.yaml",
                       help="Batch evaluation configuration file")
    
    # Input/Output paths
    parser.add_argument("--models-dir", default="models/experiments/encoder",
                       help="Directory containing trained models")
    parser.add_argument("--output-dir", default="results/experiments/evaluation",
                       help="Output directory for evaluation results")

    # Limits
    parser.add_argument("--max-models", type=int,
                       help="Maximum number of models to evaluate")
    
    # Selection
    parser.add_argument("--filter", type=str,
                       help="Filter models by pattern(s) - comma-separated (e.g., 'codebert,codet5_base')")
    parser.add_argument("--exclude", type=str,
                       help="Exclude models matching pattern(s) - comma-separated (e.g., 'codet5_large,codet5p_770m')")
    
    # Execution control
    parser.add_argument("--dry-run", action="store_true",
                       help="Show models to evaluate without running")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BatchEvaluator(args.models_dir, args.config)
    
    # Find models
    models = evaluator.find_trained_models()
    
    if not models:
        logger.error("No trained models found")
        return
    
    # Apply filters (support comma-separated patterns)
    if args.filter:
        # Split on comma and strip whitespace
        filter_patterns = [p.strip() for p in args.filter.split(',')]
        models = [m for m in models if any(pattern in m['name'] for pattern in filter_patterns)]

    # Apply exclusions (support comma-separated patterns)
    if args.exclude:
        # Split on comma and strip whitespace
        exclude_patterns = [p.strip() for p in args.exclude.split(',')]
        models = [m for m in models if not any(pattern in m['name'] for pattern in exclude_patterns)]
    
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
        filter_pattern=args.filter,
        exclude_pattern=args.exclude
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
