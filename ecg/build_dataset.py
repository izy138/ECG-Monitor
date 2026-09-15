"""Phase 1: build train/val/test beat datasets from MIT-BIH.

    python -m ecg.build_dataset                 # download (first run) + process
    python -m ecg.build_dataset --skip-download
    python -m ecg.build_dataset --val-records 116 118 207 223

Outputs (data/processed/):
    train.npz, val.npz, test.npz   arrays: X (n,200) float32, rr (n,4) float32, y (n,) int64,
                                   record (n,) str, sample (n,) int64, symbol (n,) str
    metadata.json                  label map, split records, class counts, preprocessing params
"""

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .preprocessing import bandpass, extract_window, rr_features, zscore


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def ensure_downloaded(raw_dir: Path, records) -> None:
    import wfdb

    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in records
               if not all((raw_dir / f"{r}.{ext}").exists() for ext in ("hea", "dat", "atr"))]
    if not missing:
        print(f"All {len(records)} records already in {raw_dir}")
        return
    print(f"Downloading {len(missing)} records from PhysioNet into {raw_dir} ...")
    wfdb.dl_database("mitdb", str(raw_dir), records=missing)


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_record(record_path: Path):
    """Filter the whole record, then cut beats. Returns (arrays dict, skip-reason Counter)."""
    import wfdb

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    if rec.fs != C.FS:
        raise ValueError(f"{record_path}: expected fs={C.FS}, got {rec.fs}")
    if C.LEAD not in rec.sig_name:
        raise ValueError(f"{record_path}: no {C.LEAD} lead (has {rec.sig_name})")

    # Filter the full 30-minute signal once. Filtering happens BEFORE segmentation.
    signal = bandpass(rec.p_signal[:, rec.sig_name.index(C.LEAD)])

    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample, dtype=np.int64)
    is_beat = np.fromiter((s in C.BEAT_SYMBOLS for s in symbols), dtype=bool, count=len(symbols))
    beat_samples, beat_symbols = samples[is_beat], symbols[is_beat]

    # RR computed over ALL beats, so dropping an unmapped beat doesn't fake a long interval.
    rr_all = rr_features(beat_samples)

    X, RR, y, S, SYM = [], [], [], [], []
    skipped = Counter()
    for i, (peak, sym) in enumerate(zip(beat_samples, beat_symbols)):
        cls = C.AAMI_MAP.get(sym)
        if cls is None:
            skipped["non_aami_symbol"] += 1
            continue
        if not np.all(np.isfinite(rr_all[i])):
            skipped["insufficient_rr_context"] += 1
            continue
        window = extract_window(signal, peak)
        if window is None:
            skipped["record_edge"] += 1
            continue
        window = zscore(window)
        if window is None:
            skipped["flat_signal"] += 1
            continue
        X.append(window)
        RR.append(rr_all[i])
        y.append(C.CLASS_TO_IDX[cls])
        S.append(peak)
        SYM.append(sym)

    data = {
        "X": np.stack(X).astype(np.float32) if X else np.empty((0, C.WINDOW_LEN), np.float32),
        "rr": np.stack(RR).astype(np.float32) if RR else np.empty((0, 4), np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "sample": np.asarray(S, dtype=np.int64),
        "symbol": np.asarray(SYM, dtype="<U1"),
    }
    return data, skipped


def class_counts(y) -> dict:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(C.CLASSES))
    return {c: int(counts[i]) for i, c in enumerate(C.CLASSES)}


# ---------------------------------------------------------------------------
# Validation split (patient-level, carved out of DS1)
# ---------------------------------------------------------------------------
def choose_val_records(per_record_counts: dict, candidates, sizes=(4, 5),
                       target: float = 0.2, max_frac: float = 0.5):
    """Pick whole DS1 records for validation so each class lands near `target` fraction.

    Minority classes are concentrated in a few records (most F beats sit in one record),
    so a random record pick can leave a class missing from train or val. This searches
    all 4- and 5-record combinations and rejects any that take more than `max_frac` of a
    class away from training.
    """
    candidates = sorted(candidates)
    totals = {c: sum(per_record_counts[r][c] for r in candidates) for c in C.CLASSES}
    active = [c for c in C.CLASSES if totals[c] > 0]

    best, best_score, best_fracs = None, float("inf"), None
    for k in sizes:
        for combo in itertools.combinations(candidates, k):
            fracs = {c: sum(per_record_counts[r][c] for r in combo) / totals[c] for c in active}
            if any(f > max_frac for f in fracs.values()):
                continue
            score = sum((f - target) ** 2 for f in fracs.values())
            score += sum(1.0 for f in fracs.values() if f == 0)  # strongly avoid empty classes
            if score < best_score:
                best, best_score, best_fracs = list(combo), score, fracs
    if best is None:
        raise RuntimeError("No validation split satisfies the constraints; relax max_frac.")
    return best, best_fracs


# ---------------------------------------------------------------------------
# Saving / reporting
# ---------------------------------------------------------------------------
def concat_split(per_record: dict, records) -> dict:
    parts = [per_record[r] for r in records]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "rr", "y", "sample", "symbol")}
    out["record"] = np.concatenate(
        [np.full(len(per_record[r]["y"]), r, dtype="<U3") for r in records])
    return out


def print_table(title: str, rows: dict) -> None:
    header = f"{'':>8}" + "".join(f"{c:>8}" for c in C.CLASSES) + f"{'total':>9}"
    print(f"\n{title}\n{header}")
    for name, counts in rows.items():
        total = sum(counts.values())
        print(f"{name:>8}" + "".join(f"{counts[c]:>8}" for c in C.CLASSES) + f"{total:>9}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/mitdb"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--val-records", nargs="+", default=None,
                    help="DS1 records to use for validation (default: chosen automatically)")
    args = ap.parse_args(argv)

    records = list(C.DS1) + list(C.DS2)
    if not args.skip_download:
        ensure_downloaded(args.raw_dir, records)

    per_record, counts, skipped_total = {}, {}, Counter()
    for r in records:
        data, skipped = process_record(args.raw_dir / r)
        per_record[r], counts[r] = data, class_counts(data["y"])
        skipped_total.update(skipped)
    print_table("Beats per record", counts)

    if args.val_records:
        val = sorted(args.val_records)
        bad = set(val) - set(C.DS1)
        if bad:
            raise SystemExit(f"Validation records must come from DS1; not in DS1: {sorted(bad)}")
        totals = {c: sum(counts[r][c] for r in C.DS1) for c in C.CLASSES}
        val_fracs = {c: (sum(counts[r][c] for r in val) / totals[c]) if totals[c] else 0.0
                     for c in C.CLASSES}
    else:
        val, val_fracs = choose_val_records(counts, C.DS1)
    train = [r for r in C.DS1 if r not in val]
    test = list(C.DS2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for name, recs in (("train", train), ("val", val), ("test", test)):
        split = concat_split(per_record, recs)
        np.savez_compressed(args.out_dir / f"{name}.npz", **split)
        split_counts[name] = class_counts(split["y"])
    print_table("Beats per split", split_counts)

    missing = [(s, c) for s, cc in split_counts.items() for c, n in cc.items() if n == 0]
    if missing:
        print(f"\nWARNING: empty classes in splits: {missing}")

    import wfdb
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wfdb_version": wfdb.__version__,
        "label_map": {i: c for i, c in enumerate(C.CLASSES)},
        "class_names": C.CLASS_NAMES,
        "aami_map": C.AAMI_MAP,
        "splits": {"train": train, "val": val, "test": test},
        "val_fraction_of_ds1_per_class": {c: round(f, 3) for c, f in val_fracs.items()},
        "class_counts": split_counts,
        "per_record_counts": counts,
        "skipped_beats": dict(skipped_total),
        "preprocessing": {
            "fs": C.FS, "lead": C.LEAD,
            "window_before": C.WINDOW_BEFORE, "window_after": C.WINDOW_AFTER,
            "bandpass_hz": [C.BANDPASS_LOW_HZ, C.BANDPASS_HIGH_HZ],
            "bandpass_order": C.BANDPASS_ORDER,
            "filter": "butterworth sosfiltfilt applied to full record before segmentation",
            "normalization": "per-beat z-score",
            "rr_features": list(C.RR_FEATURE_NAMES),
            "local_rr_beats": C.LOCAL_RR_BEATS,
        },
        "notes": [
            "Split follows de Chazal et al. (2004); paced records 102, 104, 107, 217 excluded.",
            "Records 201 (DS1) and 202 (DS2) are the same patient.",
            "Test set (DS2) must not be used for model selection or early stopping.",
        ],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nValidation records: {val}")
    print(f"Skipped beats: {dict(skipped_total)}")
    print(f"Saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
