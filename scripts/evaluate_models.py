#!/usr/bin/env python3
"""
Main evaluation interface for Stage 3 - Model Evaluation.

This script provides a unified interface to evaluate both generative and encoder models
on the Chef detection test dataset.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List
import time
from datetime import datetime

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

def evaluate_generative(args, config_data):
    """Evaluate generative model (CodeLLaMA with LoRA)."""
    logger.info("Evaluating generative model")
    
    # Get model-specific config for generative model
    model_config = config_data.get('models', {}).get('generative', {})
    use_4bit = model_config.get('use_4bit', True)
    max_new_tokens = model_config.get('max_new_tokens', 10)

    evaluator = GenerativeEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_4bit=use_4bit,
        max_new_tokens=max_new_tokens
    )
    
    # Use config values with CLI overrides - prefer model-specific settings
    gen_model_config = config_data.get('models', {}).get('generative', {})
    model_batch_size = gen_model_config.get('batch_size')
    batch_size = model_batch_size if model_batch_size is not None else args.batch_size
    args.batch_size = batch_size  # Update args with final value

    eval_config = config_data.get('evaluation', {})
    max_samples = args.max_samples if args.max_samples is not None else eval_config.get('max_samples')
    save_predictions = args.save_predictions if hasattr(args, 'save_predictions') else eval_config.get('save_predictions', True)
    args.save_predictions = save_predictions  # Update args with final value

    results = evaluator.evaluate(
        test_path=args.test_path,
        batch_size=batch_size,
        max_samples=max_samples,
        save_predictions=save_predictions
    )
    
    return results

def evaluate_encoder(args, config_data):
    """Evaluate encoder model (CodeBERT/CodeT5)."""
    logger.info("Evaluating encoder model")
    
    evaluator = EncoderEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # Use config values with CLI overrides - prefer model-specific settings
    enc_model_config = config_data.get('models', {}).get('encoder', {})
    model_batch_size = enc_model_config.get('batch_size')
    batch_size = model_batch_size if model_batch_size is not None else args.batch_size
    args.batch_size = batch_size  # Update args with final value

    eval_config = config_data.get('evaluation', {})
    max_samples = args.max_samples if args.max_samples is not None else eval_config.get('max_samples')
    save_predictions = args.save_predictions if hasattr(args, 'save_predictions') else eval_config.get('save_predictions', True)
    args.save_predictions = save_predictions  # Update args with final value
    
    results = evaluator.evaluate(
        test_path=args.test_path,
        batch_size=batch_size,
        max_samples=max_samples,
        save_predictions=save_predictions
    )
    
    return results

def compare_models(args, config_data):
    """Compare multiple models."""
    logger.info("Comparing models")

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
    parser = argparse.ArgumentParser(description="Evaluate Chef detection models")
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
        "--test-path",
        default="data/processed/chef_test.jsonl",
        help="Path to test data"
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
    
    # Apply config defaults where CLI args not provided
    eval_config = config_data.get('evaluation', {})

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = eval_config.get("output_dir", f"results/evaluation_{timestamp}")

    if args.test_path == "data/processed/chef_test.jsonl":  # Default value
        args.test_path = eval_config.get("test_path", args.test_path)

    if args.batch_size == 4:  # Default value
        args.batch_size = eval_config.get("batch_size", args.batch_size)

    if args.max_samples is None:
        args.max_samples = eval_config.get("max_samples")

    # Add missing arguments from config
    if not hasattr(args, 'save_predictions') or args.save_predictions is None:
        args.save_predictions = eval_config.get('save_predictions', True)
    
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
    
    # Log configuration summary
    logger.info(f"Configuration loaded from: {args.config if args.config else 'defaults'}")
    logger.info(f"Approach: {args.approach}")
    if args.approach != "compare":
        logger.info(f"Model path: {args.model_path}")
    else:
        logger.info(f"Model paths: {args.model_paths}")
    logger.info(f"Test data: {args.test_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Evaluation configuration:")
    logger.info(f"  Approach: {args.approach}")
    if args.approach != "compare":
        logger.info(f"  Model: {args.model_path}")
    else:
        logger.info(f"  Models: {args.model_paths}")
    logger.info(f"  Test data: {args.test_path}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    if args.max_samples:
        logger.info(f"  Max samples: {args.max_samples}")
    
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
        
        # Save results
        results_file = Path(args.output_dir) / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")

        # Print summary
        if "metrics" in results:
            metrics = results["metrics"]
            logger.info(f"Evaluation Summary:")
            logger.info(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"   F1-Score: {metrics.get('f1', 'N/A'):.4f}")
            logger.info(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            logger.info(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")

        elapsed_time = time.time() - start_time
        logger.info(f"Total evaluation time: {elapsed_time:.2f} seconds")

        logger.info(f"Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()