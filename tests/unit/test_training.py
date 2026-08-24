"""Model-0 sanity checks: shapes, finite loss, tiny-subset overfit,
checkpoint roundtrip and rejection paths."""

from pathlib import Path

import numpy as np
import pytest
import torch

from digit_classifier.models.baseline import MODEL_KIND, BaselineMlp
from digit_classifier.training.datasets import MNIST_MEAN, MNIST_STD, normalize_images
from digit_classifier.training.train import load_checkpoint, predict, train_model


def tiny_dataset(n_per_class: int = 6, seed: int = 0):
    """Synthetic, learnable 28x28 data: class c = bright block at column c."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for c in range(10):
        for _ in range(n_per_class):
            img = rng.integers(0, 30, size=(28, 28), dtype=np.uint8)
            img[:, c * 2 : c * 2 + 3] = 220
            images.append(img)
            labels.append(c)
    x = normalize_images(np.stack(images))
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y


def test_normalization_contract() -> None:
    img = np.zeros((2, 28, 28), dtype=np.uint8)
    img[0, :, :] = 255
    x = normalize_images(img)
    assert x.shape == (2, 1, 28, 28)
    assert x.dtype == torch.float32
    assert torch.allclose(x[0, 0, 0, 0], torch.tensor((1.0 - MNIST_MEAN) / MNIST_STD))
    assert torch.allclose(x[1, 0, 0, 0], torch.tensor((0.0 - MNIST_MEAN) / MNIST_STD))


def test_normalize_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError):
        normalize_images(np.zeros((2, 28, 28), dtype=np.float32))


def test_forward_shape_and_finite() -> None:
    model = BaselineMlp(hidden_units=16)
    x, _ = tiny_dataset(n_per_class=1)
    logits = model(x)
    assert logits.shape == (10, 10)
    assert torch.isfinite(logits).all()


def test_tiny_subset_overfit_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    x, y = tiny_dataset()
    data = {"train": (x, y), "val": (x, y)}
    model = BaselineMlp(hidden_units=32)
    result = train_model(
        model,
        data,
        model_kind=MODEL_KIND,
        run_id="test-overfit",
        seed=7,
        epochs=30,
        batch_size=16,
        learning_rate=0.01,
        artifacts_dir=tmp_path,
        extra_config={"hidden_units": 32},
    )
    # A working pipeline must be able to deliberately overfit 60 samples.
    assert result.best_val_accuracy >= 0.95
    # loss decreased over training
    import json

    meta = json.loads(result.run_metadata_path.read_text())
    losses = [e["mean_train_loss"] for e in meta["epoch_log"]]
    assert losses[-1] < losses[0]
    assert meta["checkpoint_sha256"]

    preds_before = predict(model, (x, y))
    reloaded = load_checkpoint(result.checkpoint_path, BaselineMlp(hidden_units=32), MODEL_KIND)
    preds_after = predict(reloaded, (x, y))
    assert np.array_equal(preds_before, preds_after)
    assert not reloaded.training  # eval mode after load


def test_checkpoint_rejections(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", BaselineMlp(), MODEL_KIND)

    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint at all")
    with pytest.raises(ValueError, match="malformed"):
        load_checkpoint(garbage, BaselineMlp(), MODEL_KIND)

    x, y = tiny_dataset(n_per_class=1)
    result = train_model(
        BaselineMlp(hidden_units=16),
        {"train": (x, y), "val": (x, y)},
        model_kind=MODEL_KIND,
        run_id="test-kind",
        seed=1,
        epochs=1,
        batch_size=8,
        learning_rate=0.01,
        artifacts_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="kind"):
        load_checkpoint(result.checkpoint_path, BaselineMlp(hidden_units=16), "cnn-something")
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(result.checkpoint_path, BaselineMlp(hidden_units=99), MODEL_KIND)
