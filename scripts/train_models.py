#!/usr/bin/env python3
"""
Main training interface for Stage 2 - Model Training Approaches.

This script provides a unified interface to train both generative and encoder models
for Chef detection classification.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import trainers
from trainers.generative_trainer import GenerativeTrainer
from trainers.encoder_trainer import EncoderTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_generative(args):
    """Train generative model (CodeLLaMA with LoRA)."""
    logger.info("🚀 Training generative model...")
    
    trainer = GenerativeTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Prepare datasets
    trainer.prepare_datasets(args.train_path, args.val_path)
    
    # Train the model
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    logger.info(f"✅ Generative model training completed. Model saved to {args.output_dir}")

def train_encoder(args):
    """Train encoder model (CodeBERT/CodeT5)."""
    logger.info("🚀 Training encoder model...")
    
    trainer = EncoderTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Prepare datasets
    trainer.prepare_datasets(args.train_path, args.val_path)
    
    # Train the model
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    logger.info(f"✅ Encoder model training completed. Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train Chef detection models")
    parser.add_argument(
        "--approach", 
        choices=["generative", "encoder"], 
        required=True,
        help="Training approach: generative (CodeLLaMA) or encoder (CodeBERT)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file to load defaults from (overridden by CLI)"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (defaults to recommended models for each approach)"
    )
    parser.add_argument(
        "--train-path",
        default="data/processed/chef_train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-path", 
        default="data/processed/chef_val.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (defaults to recommended values)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save model every N steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=25,
        help="Evaluate model every N steps"
    )
    
    args = parser.parse_args()
    
    # Set environment to silence tokenizers fork-parallel warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Optionally load YAML config and apply defaults where CLI not provided
    config_data: Dict[str, Any] = {}
    if args.config is not None:
        if yaml is None:
            logger.warning("PyYAML not available; --config ignored.")
        else:
            with open(args.config, "r") as f:
                config_data = yaml.safe_load(f) or {}

    # Set defaults based on approach
    if args.model_name is None:
        if args.approach == "generative":
            args.model_name = config_data.get("model_name", "codellama/CodeLlama-34b-hf")
        else:
            args.model_name = config_data.get("model_name", "microsoft/codebert-base")
    
    if args.learning_rate is None:
        if args.approach == "generative":
            args.learning_rate = config_data.get("learning_rate", 5e-5)
        else:
            args.learning_rate = config_data.get("learning_rate", 2e-5)
    
    if args.output_dir is None:
        args.output_dir = config_data.get("output_dir", f"models/{args.approach}")

    # Training hyperparameters from YAML if CLI at defaults
    # Note: CLI values override YAML. We treat parser defaults as "not explicitly set".
    parser_defaults = {
        "batch_size": 2,
        "num_epochs": 2,
        "warmup_steps": 10,
        "save_steps": 50,
        "eval_steps": 25,
    }
    # Approach-specific sensible defaults if not in YAML either
    approach_defaults = {
        "generative": {
            "batch_size": 1,
            "num_epochs": 3,
            "warmup_steps": 100,
            "save_steps": 100,
            "eval_steps": 50,
        },
        "encoder": {
            "batch_size": 8,
            "num_epochs": 3,
            "warmup_steps": 100,
            "save_steps": 100,
            "eval_steps": 50,
        },
    }

    def resolve_hp(name: str) -> int:
        cli_value = getattr(args, name)
        if cli_value == parser_defaults[name]:
            # Not explicitly set; try YAML then approach default
            return int(config_data.get(name, approach_defaults[args.approach][name]))
        return int(cli_value)

    args.batch_size = resolve_hp("batch_size")
    args.num_epochs = resolve_hp("num_epochs")
    args.warmup_steps = resolve_hp("warmup_steps")
    args.save_steps = resolve_hp("save_steps")
    args.eval_steps = resolve_hp("eval_steps")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Training configuration:")
    logger.info(f"  Approach: {args.approach}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Train data: {args.train_path}")
    logger.info(f"  Val data: {args.val_path}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Epochs: {args.num_epochs}")

    # Persist a resolved config_used.yaml/json for reproducibility
    resolved_config = {
        "approach": args.approach,
        "model_name": args.model_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", None),
    }
    with open(Path(args.output_dir) / "config_used.yaml", "w") as f:
        if yaml is not None:
            yaml.safe_dump(resolved_config, f, sort_keys=False)
        else:
            f.write(json.dumps(resolved_config, indent=2))

    # Train based on approach
    if args.approach == "generative":
        # Pull optional config-only fields for generative
        max_length = int(config_data.get("max_length", 512))
        # lora block
        lora_cfg = config_data.get("lora", {}) if isinstance(config_data.get("lora", {}), dict) else {}
        lora_r = int(lora_cfg.get("r", 16))
        lora_alpha = int(lora_cfg.get("lora_alpha", 32))
        lora_dropout = float(lora_cfg.get("lora_dropout", 0.1))
        lora_target_modules = lora_cfg.get("target_modules", None)
        # training arg toggles
        evaluation_strategy = config_data.get("evaluation_strategy", "steps")
        save_strategy = config_data.get("save_strategy", "steps")
        logging_steps = int(config_data.get("logging_steps", 10))
        load_best_model_at_end = bool(config_data.get("load_best_model_at_end", True))
        metric_for_best_model = config_data.get("metric_for_best_model", "eval_loss")
        greater_is_better = bool(config_data.get("greater_is_better", False))
        fp16 = bool(config_data.get("fp16", True))

        # Initialize trainer with LoRA params
        trainer = GenerativeTrainer(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        trainer.prepare_datasets(args.train_path, args.val_path, max_length=max_length)
        trainer.train(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            logging_steps=logging_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,
        )
        logger.info(f"✅ Generative model training completed. Model saved to {args.output_dir}")
    else:
        train_encoder(args)

if __name__ == "__main__":
    main()