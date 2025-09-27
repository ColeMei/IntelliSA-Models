"""Shared configuration defaults for encoder experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


ENCODER_GLOBAL_DEFAULTS: Dict[str, Any] = {
    "max_length": 512,
    "eval_steps": 100,
    "save_steps": 200,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "logging_steps": 25,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "train_path": "data/processed/train.jsonl",
    "val_path": "data/processed/val.jsonl",
    "fp16": True,
    "dataloader_pin_memory": True,
    "warmup_steps": 100,
}


ENCODER_EARLY_STOPPING_DEFAULTS: Dict[str, Any] = {
    "patience": 2,
    "min_delta": 0.001,
}


ENCODER_OUTPUT_DEFAULTS: Dict[str, Any] = {
    "base_dir": "models/experiments/encoder",
    "results_dir": "results/experiments/encoder",
    "naming_pattern": "{model}_{lr}_{bs}_{epochs}_{wd}",
    "configs_dir": "configs/generated",
}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""

    result = deepcopy(base)
    for key, value in (override or {}).items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def apply_encoder_defaults(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply shared encoder defaults to a stage configuration."""

    config = deepcopy(raw_config or {})

    config["global"] = _merge_dicts(
        ENCODER_GLOBAL_DEFAULTS,
        config.get("global", {}),
    )

    if "early_stopping" in config:
        config["early_stopping"] = _merge_dicts(
            ENCODER_EARLY_STOPPING_DEFAULTS,
            config.get("early_stopping", {}),
        )
    else:
        config["early_stopping"] = deepcopy(ENCODER_EARLY_STOPPING_DEFAULTS)

    config["output"] = _merge_dicts(
        ENCODER_OUTPUT_DEFAULTS,
        config.get("output", {}),
    )

    return config
