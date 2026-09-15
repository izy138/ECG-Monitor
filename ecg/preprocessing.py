"""Signal processing shared by dataset building and the streaming inference API."""

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as C


@lru_cache(maxsize=8)
def design_bandpass(fs: float = C.FS, low: float = C.BANDPASS_LOW_HZ,
                    high: float = C.BANDPASS_HIGH_HZ, order: int = C.BANDPASS_ORDER):
    # Second-order sections: the (b, a) form is numerically fragile for a 0.5 Hz
    # cutoff at 360 Hz.
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def bandpass(signal, fs: float = C.FS) -> np.ndarray:
    """Zero-phase bandpass over a LONG signal: a whole record, or a streaming buffer that
    has extra margin on both sides.

    Do not call this on a single 200-sample beat. A 0.5 Hz high-pass needs several
    seconds of context; on a 0.56 s window the filter's edge transients dominate.
    """
    return sosfiltfilt(design_bandpass(fs), np.asarray(signal, dtype=np.float64))


def extract_window(signal: np.ndarray, peak: int,
                   before: int = C.WINDOW_BEFORE, after: int = C.WINDOW_AFTER):
    """Fixed-width window around an R-peak, or None if it would run off the signal."""
    start, end = int(peak) - before, int(peak) + after
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def zscore(window: np.ndarray, eps: float = 1e-6):
    """Per-beat normalization so the model learns shape, not amplitude. None if flat."""
    std = float(np.std(window))
    if std < eps:
        return None
    return ((window - np.mean(window)) / std).astype(np.float32)


def rr_features(peaks, fs: float = C.FS, local_beats: int = C.LOCAL_RR_BEATS) -> np.ndarray:
    """Timing features for each beat, shape (n, 4):

        pre_rr_s       seconds since the previous beat
        post_rr_s      seconds until the next beat
        pre_rr_ratio   pre_rr / median of the previous `local_beats` pre-RRs
        post_rr_ratio  post_rr / that same local median

    Supraventricular beats often look almost normal; what gives them away is that they
    arrive early. The ratios make "early" patient-independent (0.6 s is early at 60 bpm
    and normal at 100 bpm). The local baseline uses only past beats, so it can be computed
    identically while streaming; post_rr costs one beat of latency.

    Undefined values (first/last beat, no history yet) are NaN.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    n = len(peaks)
    feats = np.full((n, 4), np.nan)
    if n < 2:
        return feats.astype(np.float32)

    rr = np.diff(peaks) / fs
    pre = np.full(n, np.nan)
    pre[1:] = rr
    post = np.full(n, np.nan)
    post[:-1] = rr

    local = np.full(n, np.nan)
    for i in range(2, n):
        local[i] = np.median(pre[max(1, i - local_beats):i])

    feats[:, 0] = pre
    feats[:, 1] = post
    feats[:, 2] = pre / local
    feats[:, 3] = post / local
    return feats.astype(np.float32)
