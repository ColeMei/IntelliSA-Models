#!/usr/bin/env python3
"""
Main evaluation interface for Stage 3 - Model Evaluation.

This script provides a unified interface to evaluate both generative and encoder models
on IaC security smell detection test datasets.
"""

import argparse
import logging
import os
import sys
import random
import json
from pathlib import Path
from typing import Any, Dict, List
import time
from datetime import datetime
import numpy as np

import torch

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import evaluators
from evaluation.model_comparator import ModelComparator
from evaluation.generative_evaluator import GenerativeEvaluator  
from evaluation.encoder_evaluator import EncoderEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationParameterResolver:
    """Handles parameter resolution with CLI > YAML > defaults priority."""

    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data

    def resolve_model_params(self, approach: str, args) -> Dict[str, Any]:
        """Resolve model-specific parameters."""
        model_config = self.config_data.get('models', {}).get(approach, {})
        eval_config = self.config_data.get('evaluation', {})

        # Batch size: model-specific > global eval > CLI default
        batch_size = (model_config.get('batch_size') or
                     eval_config.get('batch_size') or
                     args.batch_size)

        # Max samples: CLI > config > None
        max_samples = (args.max_samples if args.max_samples is not None else
                      eval_config.get('max_samples'))

        # Save predictions: CLI > config > True
        save_predictions = (args.save_predictions if hasattr(args, 'save_predictions') and args.save_predictions is not None else
                           eval_config.get('save_predictions', True))

        return {
            'batch_size': batch_size,
            'max_samples': max_samples,
            'save_predictions': save_predictions
        }

    def resolve_paths(self, args) -> str:
        """Resolve test path with proper fallback logic."""
        # CLI override takes precedence
        if args.test_path and args.test_path != (f"data/processed/{args.technology}/test.jsonl" if not args.combined else "data/processed/test.jsonl"):
            return args.test_path

        # Config override
        eval_config = self.config_data.get('evaluation', {})
        config_path = eval_config.get('test_path')
        if config_path:
            return config_path

        # Default based on combined/technology flag
        return "data/processed/test.jsonl" if args.combined else f"data/processed/{args.technology}/test.jsonl"

def evaluate_generative(args, config_data):
    """Evaluate generative model (CodeLLaMA with LoRA)."""
    logger.info("Evaluating generative model")

    # Get model-specific config for generative model
    model_config = config_data.get('models', {}).get('generative', {})
    use_4bit = model_config.get('use_4bit', True)
    max_new_tokens = model_config.get('max_new_tokens', 10)
    prompt_style = model_config.get('prompt_style')

    evaluator = GenerativeEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_4bit=use_4bit,
        max_new_tokens=max_new_tokens,
        prompt_style=prompt_style
    )

    # Use parameter resolver for consistent parameter handling
    param_resolver = EvaluationParameterResolver(config_data)
    params = param_resolver.resolve_model_params('generative', args)

    results = evaluator.evaluate(
        test_path=args.test_path,
        batch_size=params['batch_size'],
        max_samples=params['max_samples'],
        save_predictions=params['save_predictions']
    )

    return results

def evaluate_encoder(args, config_data):
    """Evaluate encoder model (CodeBERT/CodeT5)."""
    logger.info("Evaluating encoder model")

    # Optional threshold block from config to set envs (minimal, explicit)
    thr_cfg = (config_data.get('evaluation', {}) or {}).get('threshold') if isinstance(config_data, dict) else None
    if isinstance(thr_cfg, dict):
        mode = thr_cfg.get('mode')
        if isinstance(mode, str) and mode.strip():
            os.environ['EVAL_THRESHOLD_MODE'] = mode.strip()
        file_path = thr_cfg.get('file')
        if isinstance(file_path, str) and file_path.strip():
            os.environ['EVAL_THRESHOLD_FILE'] = file_path.strip()
        fixed_val = thr_cfg.get('fixed')
        if fixed_val is not None:
            os.environ['EVAL_THRESHOLD_FIXED'] = str(fixed_val)

    evaluator = EncoderEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir
    )

    # Use parameter resolver for consistent parameter handling
    param_resolver = EvaluationParameterResolver(config_data)
    params = param_resolver.resolve_model_params('encoder', args)

    results = evaluator.evaluate(
        test_path=args.test_path,
        batch_size=params['batch_size'],
        max_samples=params['max_samples'],
        save_predictions=params['save_predictions']
    )

    return results

def compare_models(args, config_data):
    """Compare multiple models."""

    comparator = ModelComparator(output_dir=args.output_dir)

    # Load model results - look in evaluation output directories, not model directories
    model_results = {}
    base_dir = Path(args.output_dir).parent  # Parent directory of comparison output

    for model_path in args.model_paths:
        model_name = Path(model_path).name

        # Look for results in evaluation subdirectories
        if "generative" in model_name:
            result_file = base_dir / "generative_eval" / "evaluation_results.json"
        elif "encoder" in model_name:
            result_file = base_dir / "encoder_eval" / "evaluation_results.json"
        else:
            # Fallback to model directory
            result_file = Path(model_path) / "evaluation_results.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                model_results[model_name] = json.load(f)
        else:
            logger.warning(f"No results found for {model_name} at {result_file}")

    # Generate comparison
    comparison_results = comparator.compare_models(model_results, args.test_path)

    logger.info(f"Model comparison completed.")
    return comparison_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate IaC security smell detection models")
    parser.add_argument(
        "--approach", 
        choices=["generative", "encoder", "compare"], 
        required=True,
        help="Evaluation approach: generative, encoder, or compare multiple models"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file to load defaults from"
    )
    parser.add_argument(
        "--model-path",
        required=False,
        help="Path to trained model (for single model evaluation)"
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        help="Paths to multiple trained models (for comparison)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Evaluate on combined test dataset across all technologies (default: False)"
    )
    parser.add_argument(
        "--technology",
        default="chef",
        choices=["chef", "ansible", "puppet"],
        help="IaC technology to evaluate on when not using --combined (default: chef)"
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Path to test data (auto-set based on --combined or --technology if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions to file"
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Generate confusion matrix visualization"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config_data: Dict[str, Any] = {}
    if args.config is not None:
        if yaml is None:
            logger.warning("PyYAML not available; --config ignored.")
        else:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f) or {}
    
    # Use parameter resolver for consistent parameter handling
    param_resolver = EvaluationParameterResolver(config_data)

    # Resolve paths
    args.test_path = param_resolver.resolve_paths(args)

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_config = config_data.get('evaluation', {})
        args.output_dir = eval_config.get("output_dir", f"results/evaluation_{timestamp}")
    
    # Set model paths from config if not provided via CLI
    if args.approach != "compare" and not args.model_path:
        if args.approach == "generative":
            args.model_path = config_data.get('models', {}).get('generative', {}).get('path')
        elif args.approach == "encoder":
            args.model_path = config_data.get('models', {}).get('encoder', {}).get('path')
        
        if not args.model_path:
            parser.error(f"--model-path is required for {args.approach} evaluation (not found in config)")
    
    if args.approach == "compare" and not args.model_paths:
        generative_path = config_data.get('models', {}).get('generative', {}).get('path')
        encoder_path = config_data.get('models', {}).get('encoder', {}).get('path')
        
        if generative_path and encoder_path:
            args.model_paths = [generative_path, encoder_path]
        else:
            parser.error("--model-paths is required for model comparison (not found in config)")
    
    # Deterministic seeding for evaluation repeatability
    seed_env = os.environ.get("EVAL_SEED")
    try:
        seed = int(seed_env) if seed_env is not None else 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log configuration summary
    logger.info(f"Configuration loaded from: {args.config if args.config else 'defaults'}")
    logger.info(f"Approach: {args.approach}")
    if args.approach != "compare":
        logger.info(f"Model: {args.model_path}")
    else:
        logger.info(f"Models: {args.model_paths}")
    logger.info(f"Batch size: {args.batch_size}, Test data: {args.test_path}, Output: {args.output_dir}")
    
    # Save evaluation config
    eval_config = {
        "approach": args.approach,
        "model_path": args.model_path if args.approach != "compare" else None,
        "model_paths": args.model_paths if args.approach == "compare" else None,
        "test_path": args.test_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "save_predictions": args.save_predictions,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(Path(args.output_dir) / "eval_config.yaml", "w") as f:
        if yaml is not None:
            yaml.safe_dump(eval_config, f, sort_keys=False)
        else:
            f.write(json.dumps(eval_config, indent=2))
    
    # Run evaluation
    start_time = time.time()
    
    try:
        if args.approach == "generative":
            results = evaluate_generative(args, config_data)
        elif args.approach == "encoder":
            results = evaluate_encoder(args, config_data)
        elif args.approach == "compare":
            results = compare_models(args, config_data)
        
        # Note: Results are already saved by individual evaluators or ModelComparator
        # No need to duplicate the file

        # Print summary
        if "metrics" in results:
            metrics = results["metrics"]
            logger.info(f"Results: Acc={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1', 0):.4f}, P={metrics.get('precision', 0):.4f}, R={metrics.get('recall', 0):.4f}")

        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()