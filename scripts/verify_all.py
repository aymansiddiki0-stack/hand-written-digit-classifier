"""Run the complete fast local verification suite.

Usage: uv run python scripts/verify_all.py
Exits non-zero on the first failing group and prints an honest summary.
Slow work (full training, camera benchmarks) is intentionally excluded and runs through separate commands documented in the README.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"

CHECKS: list[tuple[str, list[str], Path]] = [
    ("python-lint", ["uv", "run", "ruff", "check", "."], ROOT),
    ("python-format-check", ["uv", "run", "ruff", "format", "--check", "."], ROOT),
    ("python-tests", ["uv", "run", "pytest"], ROOT),
    ("frontend-typecheck", ["npx", "tsc", "--noEmit"], WEB),
    ("frontend-tests", ["npm", "test", "--silent"], WEB),
    ("frontend-build", ["npm", "run", "build", "--silent"], WEB),
]


def main() -> int:
    results: list[tuple[str, bool, float]] = []
    for name, cmd, cwd in CHECKS:
        rel_cwd = cwd.relative_to(ROOT) if cwd != ROOT else Path(".")
        print(f"\n=== {name}: {' '.join(cmd)} (cwd={rel_cwd})")
        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=cwd)  # noqa: S603 - fixed local commands
        elapsed = time.monotonic() - start
        ok = proc.returncode == 0
        results.append((name, ok, elapsed))
        if not ok:
            print(f"\nFAILED at group '{name}' (exit {proc.returncode}).")
            _summary(results)
            return proc.returncode or 1
    _summary(results)
    print("\nAll fast verification groups passed.")
    return 0


def _summary(results: list[tuple[str, bool, float]]) -> None:
    print("\n--- verification summary ---")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<22} {elapsed:6.1f}s")


if __name__ == "__main__":
    sys.exit(main())
