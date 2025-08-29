"""
Evaluation module for Chef detection models.
"""

from .model_comparator import ModelComparator

try:
    from .generative_evaluator import GenerativeEvaluator
except ImportError:
    GenerativeEvaluator = None

try:
    from .encoder_evaluator import EncoderEvaluator
except ImportError:
    EncoderEvaluator = None

__all__ = ['ModelComparator', 'GenerativeEvaluator', 'EncoderEvaluator']