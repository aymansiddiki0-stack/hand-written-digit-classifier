"""ONNX export and parity tests on a tiny model."""

from pathlib import Path

import pytest
import torch

from digit_classifier.inference.onnx_export import (
    export_to_onnx,
    validate_parity,
)
from digit_classifier.models.cnn import CompactCnn
from digit_classifier.training.datasets import normalize_images
from tests.unit.test_training import tiny_dataset


@pytest.fixture(scope="module")
def tiny_model_and_inputs():
    torch.manual_seed(0)
    model = CompactCnn(dropout_conv=0.0, dropout_fc=0.0).eval()
    x, _ = tiny_dataset(n_per_class=4)
    return model, x


def test_export_and_full_parity(tmp_path: Path, tiny_model_and_inputs) -> None:
    model, x = tiny_model_and_inputs
    onnx_path = export_to_onnx(model, tmp_path / "tiny.onnx")
    assert onnx_path.stat().st_size > 0
    report = validate_parity(model, onnx_path, x, report_path=tmp_path / "parity.json")
    assert report["n_samples"] == x.shape[0]
    assert report["full_agreement"] is True
    assert report["within_tolerance"] is True
    assert (tmp_path / "parity.json").is_file()


def test_parity_detects_mismatched_model(tmp_path: Path, tiny_model_and_inputs) -> None:
    model, x = tiny_model_and_inputs
    onnx_path = export_to_onnx(model, tmp_path / "a.onnx")
    torch.manual_seed(123)
    different = CompactCnn(dropout_conv=0.0, dropout_fc=0.0).eval()
    report = validate_parity(different, onnx_path, x)
    # A different network must not silently pass parity.
    assert report["max_prob_diff"] > report["tolerance"]
    assert report["within_tolerance"] is False


def test_parity_rejects_bad_input_shape(tmp_path: Path, tiny_model_and_inputs) -> None:
    model, _ = tiny_model_and_inputs
    onnx_path = export_to_onnx(model, tmp_path / "b.onnx")
    import numpy as np

    bad = normalize_images(np.zeros((2, 28, 28), dtype=np.uint8)).squeeze(1)  # [2,28,28]
    with pytest.raises(ValueError, match="expected"):
        validate_parity(model, onnx_path, bad)
