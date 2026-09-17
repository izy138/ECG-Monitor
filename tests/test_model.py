"""Smoke tests for the beat classifier."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecg import config as C  # noqa: E402
from ecg.model import BeatCNN  # noqa: E402


def test_beat_cnn_forward():
    model = BeatCNN()
    batch = 8
    x = torch.randn(batch, 1, C.WINDOW_LEN)
    rr = torch.randn(batch, len(C.RR_FEATURE_NAMES))
    logits = model(x, rr)
    assert logits.shape == (batch, len(C.CLASSES))


def test_shift_window():
    from ecg.train import shift_window

    w = np.arange(C.WINDOW_LEN, dtype=np.float32)
    shifted = shift_window(w, 3)
    assert shifted[0] == 0
    assert np.allclose(shifted[3:], w[:-3])
