# Handwritten Digit Classifier

A local, camera-enabled handwritten digit recognition app. The plan: open a
local web app, show one handwritten digit (0-9) to your webcam inside a
guide box, and get a prediction back with a confidence score and latency —
or an honest "uncertain" state instead of a guess.

## Planned stack

- **Data + model:** Python, PyTorch, ONNX (trained on MNIST, evaluated on
  real camera captures later)
- **API:** FastAPI, serving a single local `/v1/predictions` endpoint
- **Web app:** React + TypeScript + Vite, talking to the API over a local
  dev-server proxy

## Status

Just starting. Repo skeleton and the shared config boundary go in first, then
the data pipeline, then the models, then the API and camera UI.

## Quick start (native local processes)

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+.

```sh
# 1. Python environment
uv sync

# 2. Frontend dependencies
cd apps/web && npm install && cd ../..

# 3. Start the API (terminal 1)
uv run python -m services.api.main
# -> http://127.0.0.1:8000/health returns {"status":"ok",...}
# -> http://127.0.0.1:8000/ready returns 503 until a model is loaded (expected)

# 4. Start the web app (terminal 2)
cd apps/web && npm run dev
# -> open the printed local URL; the shell shows the live API status
```

## Verification

One command runs the fast verification suite (lint, format check, Python
tests, frontend typecheck, frontend tests, frontend build):

```sh
uv run python scripts/verify_all.py        # all 6 groups pass (last run)
uv run pytest                              # 41 passed, 1 skipped (last run)
```

The expected skip is the selected-model integration path; `current.json` is
not created until model selection lands.

## Verified environment (setup notes)

Recorded from the machine this milestone was built and tested on:

- OS: Linux x86_64 (kernel 6.18, Ubuntu-based)
- CPU: 1 × Intel Xeon @ 2.10 GHz — **no GPU**; all training targets CPU
- RAM: 3.9 GiB
- Python 3.12.3, uv 0.11.7, Node.js 22, npm 10 (pnpm not installed — npm is
  used instead to avoid duplicating package managers)

## Team

- **Ayman Siddiki** — data pipeline, models, training, evaluation, ONNX
- **Mahib Nasif** — API, web app, camera integration
