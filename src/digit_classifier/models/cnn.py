"""Compact CNN for 28x28 grayscale digits.

Two convolutional blocks + two dense layers. Small enough to train on a
single CPU in minutes and easy to explain; no pretrained backbone.
"""

from __future__ import annotations

import torch
from torch import nn

MODEL_KIND = "compact-cnn"


class CompactCnn(nn.Module):
    def __init__(self, dropout_conv: float = 0.25, dropout_fc: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28 -> 26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),  # 26 -> 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout(dropout_conv),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
