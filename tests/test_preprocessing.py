import numpy as np
import pytest

from ecg import config as C
from ecg.preprocessing import bandpass, extract_window, rr_features, zscore


# --- config ---------------------------------------------------------------
def test_record_lists_are_consistent():
    assert len(C.ALL_RECORDS) == 48
    assert "211" not in C.ALL_RECORDS
    assert len(C.DS1) == len(C.DS2) == 22
    assert not set(C.DS1) & set(C.DS2)
    assert set(C.DS1) | set(C.DS2) | set(C.PACED_RECORDS) == set(C.ALL_RECORDS)


def test_aami_map_targets_known_classes():
    assert set(C.AAMI_MAP.values()) == set(C.CLASSES)
    assert set(C.AAMI_MAP) <= C.BEAT_SYMBOLS


# --- bandpass -------------------------------------------------------------
def _sine(freq, seconds=60, fs=C.FS):
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * freq * t)


def test_bandpass_removes_baseline_wander():
    assert np.std(bandpass(_sine(0.1))) < 0.05


def test_bandpass_keeps_qrs_band():
    assert np.std(bandpass(_sine(10))) / np.std(_sine(10)) > 0.95


def test_bandpass_attenuates_powerline():
    assert np.std(bandpass(_sine(60))) / np.std(_sine(60)) < 0.2


# --- windowing / normalization -------------------------------------------
def test_extract_window_shape_and_edges():
    sig = np.arange(1000.0)
    w = extract_window(sig, 500)
    assert w.shape == (C.WINDOW_LEN,)
    assert w[C.WINDOW_BEFORE] == 500
    assert extract_window(sig, C.WINDOW_BEFORE - 1) is None
    assert extract_window(sig, len(sig) - C.WINDOW_AFTER + 1) is None
    assert extract_window(sig, len(sig) - C.WINDOW_AFTER) is not None


def test_zscore():
    z = zscore(np.random.default_rng(0).normal(5, 3, 200))
    assert z.dtype == np.float32
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-5
    assert zscore(np.full(200, 2.0)) is None


# --- RR features ----------------------------------------------------------
def test_rr_features_flag_premature_beat():
    # Regular 1.0 s rhythm, then a beat arriving at 0.6 s, then a compensatory 1.4 s pause.
    intervals = [1.0] * 12 + [0.6, 1.4] + [1.0] * 5
    peaks = np.concatenate([[0], np.cumsum(intervals)]) * C.FS
    f = rr_features(peaks)

    assert f.shape == (len(peaks), 4)
    assert np.all(np.isnan(f[0, [0, 2, 3]]))       # no previous beat
    assert np.isnan(f[-1, 1])                       # no next beat
    assert np.all(np.isnan(f[1, 2:]))               # no history for local median yet

    early = 13                                      # the premature beat
    assert f[early, 0] == pytest.approx(0.6)
    assert f[early, 1] == pytest.approx(1.4)
    assert f[early, 2] == pytest.approx(0.6)        # 0.6 / median(1.0)
    assert f[early - 1, 2] == pytest.approx(1.0)    # normal beat before it


def test_rr_features_short_input():
    assert rr_features([100]).shape == (1, 4)
    assert rr_features([]).shape == (0, 4)
