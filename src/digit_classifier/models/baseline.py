"""Baseline model: a small MLP.

Purpose: validate the complete pipeline, provide a transparent benchmark for
the CNN, and catch data/metric errors early. Deliberately simple.
"""

from __future__ import annotations

import torch
from torch import nn

MODEL_KIND = "baseline-mlp"


class BaselineMlp(nn.Module):
    def __init__(self, hidden_units: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
