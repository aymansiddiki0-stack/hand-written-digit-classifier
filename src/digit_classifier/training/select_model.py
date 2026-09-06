"""Select a trained model as the current serving artifact.

Usage: uv run python -m digit_classifier.training.select_model <run_id>
Applies the MNIST release gate (test accuracy >= 0.99, macro F1 >= 0.990)
and refuses selection when the gate fails. Writes artifacts/models/current.json.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from digit_classifier.config import find_project_root

GATE_ACCURACY = 0.99
GATE_MACRO_F1 = 0.990
PER_CLASS_RECALL_FLAG = 0.98
CURRENT_MODEL_FILE = "current.json"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: select_model <run_id>")
        return 2
    run_id = sys.argv[1]
    root = find_project_root()
    metrics_path = root / "artifacts" / "metrics" / f"{run_id}_metrics.json"
    meta_path = root / "artifacts" / "run_metadata" / f"{run_id}.json"
    if not metrics_path.is_file() or not meta_path.is_file():
        print(f"missing artifacts for run {run_id}")
        return 2

    metrics = json.loads(metrics_path.read_text())
    meta = json.loads(meta_path.read_text())
    test = metrics["test"]
    acc, f1 = test["accuracy"], test["macro_f1"]
    print(f"candidate {run_id}: test accuracy {acc:.4f}, macro F1 {f1:.4f}")

    low = [c for c in test["per_class"] if c["recall"] < PER_CLASS_RECALL_FLAG]
    for c in low:
        print(f"  NOTE: digit {c['digit']} recall {c['recall']:.4f} < {PER_CLASS_RECALL_FLAG}")

    if acc < GATE_ACCURACY or f1 < GATE_MACRO_F1:
        print(
            f"REFUSED: release gate requires accuracy >= {GATE_ACCURACY} "
            f"and macro F1 >= {GATE_MACRO_F1}"
        )
        return 1

    payload = {
        "model_id": run_id,
        "model_kind": metrics["model_kind"],
        "checkpoint_file": meta["checkpoint_file"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "model_config": meta.get("model_config", {}),
        "selected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_accuracy": acc,
    }
    path = root / "artifacts" / "models" / CURRENT_MODEL_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selected: {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
