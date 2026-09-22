"""End-to-end tests against synthetic WFDB records, so the pipeline can be verified
without downloading MIT-BIH."""

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

from ecg import config as C  # noqa: E402
from ecg.build_dataset import choose_val_records, class_counts, process_record  # noqa: E402


def _synthetic_ecg(peaks, n_samples, fs=C.FS, seed=0):
    """Gaussian 'QRS' spikes plus baseline wander and noise. Crude, but enough to test plumbing."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    sig = 0.3 * np.sin(2 * np.pi * 0.2 * t / fs) + 0.02 * rng.normal(size=n_samples)
    for p in peaks:
        sig += 1.0 * np.exp(-0.5 * ((t - p) / 4) ** 2)
    return sig


@pytest.fixture
def synthetic_record(tmp_path):
    n = 30 * C.FS
    peaks = list(range(50, n - 40, C.FS))           # first and last beats sit near the edges
    symbols = ["N"] * len(peaks)
    symbols[8], symbols[12], symbols[15], symbols[18] = "A", "V", "F", "L"
    symbols[20] = "?"                                # beat, but not an AAMI class

    mlii = _synthetic_ecg(peaks, n)
    v1 = _synthetic_ecg(peaks, n, seed=1)
    wfdb.wrsamp("900", fs=C.FS, units=["mV", "mV"], sig_name=["MLII", "V1"],
                p_signal=np.column_stack([mlii, v1]), fmt=["212", "212"],
                write_dir=str(tmp_path))

    # Interleave a non-beat noise annotation; it must not break RR intervals.
    ann_samples = peaks[:5] + [peaks[4] + 100] + peaks[5:]
    ann_symbols = symbols[:5] + ["~"] + symbols[5:]
    wfdb.wrann("900", "atr", sample=np.array(ann_samples), symbol=ann_symbols,
               write_dir=str(tmp_path))
    return tmp_path / "900", peaks, symbols


def test_process_record(synthetic_record):
    path, peaks, symbols = synthetic_record
    data, skipped = process_record(path)

    n = len(data["y"])
    assert data["X"].shape == (n, C.WINDOW_LEN)
    assert data["rr"].shape == (n, 4)
    assert np.all(np.isfinite(data["rr"]))
    assert np.allclose(data["X"].mean(1), 0, atol=1e-4)

    # The '~' annotation was ignored, so RR stays ~1 s for normal beats.
    normal = data["symbol"] == "N"
    assert np.allclose(data["rr"][normal, 0], 1.0, atol=1e-2)

    assert set(data["symbol"]) == {"N", "A", "V", "F", "L"}
    assert class_counts(data["y"])["S"] == 1
    assert class_counts(data["y"])["V"] == 1
    assert class_counts(data["y"])["F"] == 1
    assert skipped["non_aami_symbol"] == 1          # the '?'
    assert skipped["insufficient_rr_context"] >= 2  # first two beats + last beat
    # R-peak sits at the window center index
    assert np.all(np.argmax(data["X"], axis=1) == C.WINDOW_BEFORE)


@pytest.fixture
def synthetic_incart_record(tmp_path):
    """Same shape as `synthetic_record`, but at INCART's native fs (257 Hz) and lead name
    ("II"), to exercise process_record's resample + lead-preference code paths without a
    real download."""
    fs = 257
    n = 30 * fs
    peaks = list(range(50, n - 40, fs))
    symbols = ["N"] * len(peaks)
    symbols[8], symbols[12] = "A", "V"

    lead_ii = _synthetic_ecg(peaks, n, fs=fs)
    lead_i = _synthetic_ecg(peaks, n, fs=fs, seed=1)
    wfdb.wrsamp("901", fs=fs, units=["mV", "mV"], sig_name=["I", "II"],
                p_signal=np.column_stack([lead_i, lead_ii]), fmt=["212", "212"],
                write_dir=str(tmp_path))
    wfdb.wrann("901", "atr", sample=np.array(peaks), symbol=symbols, write_dir=str(tmp_path))
    return tmp_path / "901"


def test_process_record_resamples_non_mitdb_source(synthetic_incart_record):
    data, skipped = process_record(synthetic_incart_record, db="incartdb")

    n = len(data["y"])
    assert n > 0
    assert data["X"].shape == (n, C.WINDOW_LEN)          # window is in TARGET samples (C.FS)
    assert np.all(np.isfinite(data["rr"]))
    # Beats were built 1s apart at the native rate; after resampling to C.FS, RR should still
    # read ~1s in seconds (rr_features divides by C.FS), not distorted by the resample.
    normal = data["symbol"] == "N"
    assert np.allclose(data["rr"][normal, 0], 1.0, atol=0.05)
    assert np.all(np.argmax(data["X"], axis=1) == C.WINDOW_BEFORE)


def test_process_record_lead_not_found_raises(synthetic_incart_record):
    # The record only has leads I/II; asking for svdb's "ECG1" preference must fail loudly,
    # not silently fall back to some other channel.
    with pytest.raises(ValueError, match="ECG1"):
        process_record(synthetic_incart_record, db="svdb")


def test_process_record_mitdb_rejects_non_360_fs(tmp_path):
    # mitdb's branch must never resample, even given an otherwise-valid MLII-named lead -- a
    # non-360 fs record routed through db="mitdb" is a bug upstream and must raise on the fs
    # check specifically, not silently resample. (Uses the right lead name so the fs check,
    # not the lead check, is what's being exercised here.)
    fs = 257
    n = 5 * fs
    peaks = [50, 50 + fs]
    mlii = _synthetic_ecg(peaks, n, fs=fs)
    v1 = _synthetic_ecg(peaks, n, fs=fs, seed=1)
    wfdb.wrsamp("902", fs=fs, units=["mV", "mV"], sig_name=["MLII", "V1"],
                p_signal=np.column_stack([mlii, v1]), fmt=["212", "212"], write_dir=str(tmp_path))
    wfdb.wrann("902", "atr", sample=np.array(peaks), symbol=["N", "N"], write_dir=str(tmp_path))

    with pytest.raises(ValueError, match="fs=360"):
        process_record(tmp_path / "902", db="mitdb")


def test_choose_val_records_large_pool_uses_sampling_fallback():
    # 40 candidates with sizes that would be exhaustively infeasible at INCART's real scale
    # (75 records, sizes ~14-16) -- here sized down so the test itself stays fast, but still
    # forces the random-sampling fallback path (max_exhaustive deliberately tiny).
    counts = {f"r{i}": {"N": 100, "S": i % 4, "V": (i * 2) % 5, "F": 1 if i % 7 == 0 else 0}
              for i in range(40)}
    val, fracs = choose_val_records(counts, list(counts), sizes=(8,), target=0.2,
                                    max_exhaustive=10, sample_combos=200, seed=1)
    assert len(val) == 8
    assert all(0 <= f <= 0.5 for f in fracs.values())


def test_choose_val_records_respects_constraints():
    counts = {f"r{i}": {"N": 1000, "S": 10 * (i % 3), "V": 50 * (i % 2), "F": 0}
              for i in range(10)}
    counts["r0"]["F"] = 400                         # one record holds almost all F beats
    counts["r5"]["F"] = 20
    counts["r7"]["F"] = 20
    val, fracs = choose_val_records(counts, list(counts))

    assert "r0" not in val                          # would strip F from training
    assert all(0 < f <= 0.5 for f in fracs.values())
    assert 4 <= len(val) <= 5
