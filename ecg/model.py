"""1D CNN beat classifier with RR-interval features in the head."""

import torch
import torch.nn as nn

from . import config as C


class BeatCNN(nn.Module):
    """CNN over a single beat waveform, with RR timing features concatenated before classification."""

    def __init__(self, n_classes: int = len(C.CLASSES), n_rr: int = len(C.RR_FEATURE_NAMES),
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + n_rr, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, window); rr: (batch, n_rr)
        h = self.features(x).flatten(1)
        return self.head(torch.cat([h, rr], dim=1))
