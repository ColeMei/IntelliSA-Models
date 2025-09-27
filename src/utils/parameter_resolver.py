"""Shared parameter resolution utilities for training/evaluation scripts."""

from __future__ import annotations

from typing import Any, Dict


class ParameterResolver:
    """Resolve parameters using priority: CLI > YAML > approach defaults > parser defaults."""

    def __init__(self, approach: str, config_data: Dict[str, Any]):
        self.approach = approach
        self.config_data = config_data

        # Approach-specific defaults mirror `train_models.py` historical behaviour.
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
        """Resolve a single parameter value."""

        if cli_value != parser_default:
            return cli_value

        yaml_value = self.config_data.get(param_name)
        if yaml_value is not None:
            return yaml_value

        approach_defaults = self.approach_defaults.get(self.approach, {})
        if param_name in approach_defaults:
            return approach_defaults[param_name]

        return parser_default
