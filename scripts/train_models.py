#!/usr/bin/env python3
"""
Main training interface for Stage 2 - Model Training Approaches.

This script provides a unified interface to train both generative and encoder models
for IaC security smell detection classification.
"""

import argparse
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, Optional
import json
import numpy as np

import torch

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
from utils.config_defaults import apply_encoder_defaults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterResolver:
    """Handles parameter resolution with CLI > YAML > approach defaults > parser defaults priority."""

    def __init__(self, approach: str, config_data: Dict[str, Any]):
        self.approach = approach
        self.config_data = config_data

        # Approach-specific defaults
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

def train_generative(args, config_data: Optional[Dict[str, Any]] = None):
    """Train generative model (CodeLLaMA with LoRA)."""
    logger.info("Training generative model")

    # Extract LoRA and optimization parameters from config
    lora_config = config_data.get("lora", {}) if config_data else {}
    use_4bit = config_data.get("use_4bit", True) if config_data else True
    gradient_accumulation_steps = config_data.get("gradient_accumulation_steps", 4) if config_data else 4
    weight_decay = config_data.get("weight_decay", 0.01) if config_data else 0.01

    # Extract LoRA parameters
    lora_r = lora_config.get("r", 16)
    lora_alpha = lora_config.get("lora_alpha", 32)
    lora_dropout = lora_config.get("lora_dropout", 0.1)
    lora_target_modules = lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

    prompt_style = None
    if config_data:
        prompt_style = config_data.get("prompt_style")

    trainer = GenerativeTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_4bit=use_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        prompt_style=prompt_style
    )

    # Prepare datasets
    max_length = 512
    if config_data:
        max_length = int(config_data.get("max_length", max_length))
    trainer.prepare_datasets(args.train_path, args.val_path, max_length=max_length)

    # Train the model with all parameters
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        evaluation_strategy=(config_data.get("evaluation_strategy", "steps") if config_data else "steps"),
        save_strategy=(config_data.get("save_strategy", "steps") if config_data else "steps"),
        logging_steps=(int(config_data.get("logging_steps", 10)) if config_data else 10),
        load_best_model_at_end=(bool(config_data.get("load_best_model_at_end", True)) if config_data else True),
        metric_for_best_model=(config_data.get("metric_for_best_model", "eval_loss") if config_data else "eval_loss"),
        greater_is_better=(bool(config_data.get("greater_is_better", False)) if config_data else False)
    )

    logger.info(f" Generative model training completed. Model saved to {args.output_dir}")

def train_encoder(args, config_data: Optional[Dict[str, Any]] = None):
    """Train encoder model (CodeBERT/CodeT5)."""
    logger.info("Training encoder model")

    trainer = EncoderTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Set threshold sweep config if available
    if config_data and 'threshold_sweep' in config_data:
        trainer.threshold_sweep_config = config_data['threshold_sweep']

    # Prepare datasets
    max_length = int(config_data.get("max_length", 512)) if config_data else 512
    trainer.prepare_datasets(args.train_path, args.val_path, max_length=max_length)

    # Extract additional parameters from config if available
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    early_stopping_patience = None
    early_stopping_min_delta = 0.001
    early_stopping_metric = "f1"
    early_stopping_mode = "max"
    lr_scheduler_type = "linear"
    fp16 = True
    dataloader_pin_memory = True

    if config_data:
        weight_decay = config_data.get('weight_decay', weight_decay)
        gradient_accumulation_steps = config_data.get('gradient_accumulation_steps', gradient_accumulation_steps)
        gradient_checkpointing = config_data.get('gradient_checkpointing', gradient_checkpointing)
        early_stopping_patience = config_data.get('early_stopping_patience', early_stopping_patience)
        early_stopping_min_delta = config_data.get('early_stopping_min_delta', early_stopping_min_delta)
        early_stopping_metric = config_data.get('early_stopping_metric', early_stopping_metric)
        early_stopping_mode = config_data.get('early_stopping_mode', early_stopping_mode)
        lr_scheduler_type = config_data.get('lr_scheduler_type', lr_scheduler_type)
        fp16 = config_data.get('fp16', fp16)
        dataloader_pin_memory = config_data.get('dataloader_pin_memory', dataloader_pin_memory)

    # Train the model with optimized parameters
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
        lr_scheduler_type=lr_scheduler_type,
        fp16=fp16,
        dataloader_pin_memory=dataloader_pin_memory
    )

    logger.info(f" Encoder model training completed. Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train IaC security smell detection models")
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
        "--combined",
        action="store_true",
        help="Train on combined dataset across all technologies (default: False)"
    )
    parser.add_argument(
        "--technology",
        default="chef",
        choices=["chef", "ansible", "puppet"],
        help="IaC technology to train on when not using --combined (default: chef)"
    )
    parser.add_argument(
        "--train-path",
        default=None,
        help="Path to training data (auto-set based on --combined or --technology if not provided)"
    )
    parser.add_argument(
        "--val-path",
        default=None,
        help="Path to validation data (auto-set based on --combined or --technology if not provided)"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()

    # Set default paths based on combined flag or technology
    if args.train_path is None:
        if args.combined:
            args.train_path = "data/processed/train.jsonl"
        else:
            args.train_path = f"data/processed/{args.technology}/train.jsonl"
    if args.val_path is None:
        if args.combined:
            args.val_path = "data/processed/val.jsonl"
        else:
            args.val_path = f"data/processed/{args.technology}/val.jsonl"

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
            if args.approach == "encoder":
                config_data = apply_encoder_defaults(config_data)

    # Set defaults based on approach
    if args.model_name is None:
        if args.approach == "generative":
            args.model_name = config_data.get("model_name", "codellama/CodeLlama-34b-hf")
        else:
            args.model_name = config_data.get("model_name", "microsoft/codebert-base")
    
    if args.learning_rate is None:
        if args.approach == "generative":
            args.learning_rate = float(config_data.get("learning_rate", 5e-5))
        else:
            args.learning_rate = float(config_data.get("learning_rate", 2e-5))
    
    if args.output_dir is None:
        args.output_dir = config_data.get("output_dir", f"models/{args.approach}")

    # Standardize parameter resolution: CLI > YAML > approach defaults > parser defaults
    param_resolver = ParameterResolver(args.approach, config_data)

    args.batch_size = param_resolver.resolve("batch_size", getattr(args, "batch_size"), 2)
    args.num_epochs = param_resolver.resolve("num_epochs", getattr(args, "num_epochs"), 2)
    args.warmup_steps = param_resolver.resolve("warmup_steps", getattr(args, "warmup_steps"), 10)
    args.save_steps = param_resolver.resolve("save_steps", getattr(args, "save_steps"), 50)
    args.eval_steps = param_resolver.resolve("eval_steps", getattr(args, "eval_steps"), 25)
    args.seed = param_resolver.resolve("seed", getattr(args, "seed"), 42)
    
    # Global deterministic seeding to reduce variance
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Deterministic algorithms for eval stability (training may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as _:
        pass

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
        "seed": args.seed,
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", None),
    }
    with open(Path(args.output_dir) / "config_used.yaml", "w") as f:
        if yaml is not None:
            yaml.safe_dump(resolved_config, f, sort_keys=False)
        else:
            f.write(json.dumps(resolved_config, indent=2))

    # Train based on approach
    if args.approach == "generative":
        train_generative(args, config_data)
    else:
        train_encoder(args, config_data)

if __name__ == "__main__":
    main()
