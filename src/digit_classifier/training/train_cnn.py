"""Train and evaluate the compact CNN on the recorded MNIST split.

Usage: uv run python -m digit_classifier.training.train_cnn
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from digit_classifier.config import find_project_root, load_config
from digit_classifier.models.cnn import MODEL_KIND, CompactCnn
from digit_classifier.training.datasets import load_mnist_tensors
from digit_classifier.training.train import evaluate_split, sha256_file, train_model


def main() -> None:
    root = find_project_root()
    cfg = load_config()
    interim = root / cfg.data.interim_dir
    processed = root / cfg.data.processed_dir

    data = load_mnist_tensors(interim, processed)
    run_id = f"cnn-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    model = CompactCnn(dropout_conv=cfg.cnn.dropout_conv, dropout_fc=cfg.cnn.dropout_fc)

    dataset_checksums = {
        "split_indices": sha256_file(processed / "mnist_split_indices.npz"),
        "mnist_manifest": sha256_file(interim / "mnist_manifest.json"),
    }

    result = train_model(
        model,
        data,
        model_kind=MODEL_KIND,
        run_id=run_id,
        seed=cfg.training.seed,
        epochs=cfg.cnn.epochs,
        batch_size=cfg.cnn.batch_size,
        learning_rate=cfg.cnn.learning_rate,
        artifacts_dir=root / "artifacts",
        params_path=root / "params.yaml",
        dataset_checksums=dataset_checksums,
        extra_config={
            "dropout_conv": cfg.cnn.dropout_conv,
            "dropout_fc": cfg.cnn.dropout_fc,
        },
    )

    val_metrics = evaluate_split(model, data["val"])
    test_metrics = evaluate_split(model, data["test"])

    metrics_dir = root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_kind": MODEL_KIND,
                "best_epoch": result.best_epoch,
                "validation": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"run_id: {run_id}")
    print(f"best epoch: {result.best_epoch}  best val acc: {result.best_val_accuracy:.4f}")
    print(
        f"test accuracy: {test_metrics['accuracy']:.4f}  macro F1: {test_metrics['macro_f1']:.4f}"
    )
    low = [c for c in test_metrics["per_class"] if c["recall"] < 0.98]
    print(
        "classes with recall < 0.98:", [(c["digit"], round(c["recall"], 4)) for c in low] or "none"
    )
    print(f"metrics artifact: {metrics_path.relative_to(root)}")


if __name__ == "__main__":
    main()
