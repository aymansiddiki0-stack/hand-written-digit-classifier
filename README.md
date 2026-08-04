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

## Team

- **Ayman Siddiki** — data pipeline, models, training, evaluation, ONNX
- **Mahib Nasif** — API, web app, camera integration
