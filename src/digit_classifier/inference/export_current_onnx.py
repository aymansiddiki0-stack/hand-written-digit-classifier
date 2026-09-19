"""Export the current model to ONNX and validate PyTorch parity.

Usage: uv run python -m digit_classifier.inference.export_current_onnx
Parity set: >= 1000 held-out official MNIST test samples.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from digit_classifier.config import find_project_root, load_config
from digit_classifier.inference.onnx_export import export_to_onnx, validate_parity
from digit_classifier.inference.predictor import load_current_model
from digit_classifier.training.datasets import normalize_images

N_PARITY_SAMPLES = 2000


def main() -> int:
    root = find_project_root()
    cfg = load_config()
    models_dir = root / "artifacts" / "models"
    predictor = load_current_model(models_dir)
    model = predictor._model  # noqa: SLF001 - internal DE tooling

    test_images = np.load(root / cfg.data.interim_dir / "mnist_test_images.npy")
    inputs = normalize_images(test_images[:N_PARITY_SAMPLES])

    onnx_path = models_dir / f"{predictor.model_id}.onnx"
    export_to_onnx(model, onnx_path)
    report_path = root / "artifacts" / "reports" / f"onnx_parity_{predictor.model_id}.json"
    report = validate_parity(model, onnx_path, inputs, report_path=report_path)

    print(json.dumps(report, indent=2))
    if not (report["full_agreement"] and report["within_tolerance"]):
        print("PARITY FAILED")
        return 1
    print(f"onnx artifact: {onnx_path.relative_to(root)}")
    print(f"parity report: {report_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
