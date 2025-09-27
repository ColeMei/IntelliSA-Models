import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.trainers.encoder_trainer import IacDetectionDataset, compute_metrics
from src.utils.parameter_resolver import ParameterResolver
from src.evaluation.encoder_evaluator import EncoderEvaluator


class DummyTokenizer:
    pad_token = "[PAD]"

    def __call__(self, text, truncation, max_length, padding, return_tensors="pt"):
        del text, truncation, max_length, padding, return_tensors
        return {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def _write_jsonl(samples):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    return path


def test_iac_detection_dataset_maps_labels():
    samples = [
        {"content": "foo", "label": "TP", "smell": "hard_coded_secret"},
        {"content": "bar", "label": "FP", "smell": "weak_crypto"},
    ]
    data_path = _write_jsonl(samples)
    try:
        dataset = IacDetectionDataset(data_path, DummyTokenizer(), max_length=16)

        assert len(dataset) == 2
        first = dataset[0]
        second = dataset[1]
        assert first["labels"].item() == 1
        assert second["labels"].item() == 0
    finally:
        os.remove(data_path)


def test_compute_metrics_handles_numpy_predictions():
    preds = SimpleNamespace(
        predictions=np.array([[0.1, 0.9], [0.8, 0.2]]),
        label_ids=np.array([1, 0]),
    )
    metrics = compute_metrics(preds)
    assert pytest.approx(metrics["f1"], rel=1e-4) == 1.0
    assert pytest.approx(metrics["accuracy"], rel=1e-4) == 1.0


def test_parameter_resolver_priority_cli_over_yaml():
    config = {"batch_size": 4}
    resolver = ParameterResolver("encoder", config)
    value = resolver.resolve("batch_size", cli_value=8, parser_default=2)
    assert value == 8


def test_parameter_resolver_falls_back_to_yaml():
    config = {"batch_size": 4}
    resolver = ParameterResolver("encoder", config)
    value = resolver.resolve("batch_size", cli_value=2, parser_default=2)
    assert value == 4


def test_parameter_resolver_uses_approach_default():
    resolver = ParameterResolver("encoder", {})
    value = resolver.resolve("batch_size", cli_value=2, parser_default=2)
    assert value == 8


def test_resolve_threshold_missing_file(monkeypatch):
    evaluator = object.__new__(EncoderEvaluator)
    monkeypatch.setenv("EVAL_THRESHOLD_MODE", "file")
    monkeypatch.setenv("EVAL_THRESHOLD_FILE", "/tmp/does_not_exist.json")

    with pytest.raises(RuntimeError) as exc:
        evaluator._resolve_threshold()

    assert "codet5p_220m_final_sweep_latest" in str(exc.value)


def test_resolve_threshold_reads_value(monkeypatch):
    evaluator = object.__new__(EncoderEvaluator)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"best_threshold": 0.42}, handle)
        path = handle.name

    monkeypatch.setenv("EVAL_THRESHOLD_MODE", "file")
    monkeypatch.setenv("EVAL_THRESHOLD_FILE", path)

    try:
        threshold = evaluator._resolve_threshold()
        assert pytest.approx(threshold, rel=1e-6) == 0.42
    finally:
        os.unlink(path)
