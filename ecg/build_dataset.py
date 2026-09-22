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
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .preprocessing import bandpass, extract_window, resample_record, rr_features, zscore


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def ensure_downloaded(raw_dir: Path, records, db: str = "mitdb") -> list[str]:
    """Download any of `records` not already cached in raw_dir. Returns the records that are
    actually present afterward (== `records` for mitdb, which has no gaps; may be a subset
    for incartdb/svdb, where a record can be genuinely missing/corrupt on PhysioNet -- e.g.
    svdb's 830 404s -- and any OTHER gap is handled the same way, not special-cased).
    """
    import wfdb

    raw_dir.mkdir(parents=True, exist_ok=True)
    have = lambda r: all((raw_dir / f"{r}.{ext}").exists() for ext in ("hea", "dat", "atr"))
    missing = [r for r in records if not have(r)]
    if missing:
        print(f"Downloading {len(missing)} records from {db} into {raw_dir} ...")
        try:
            wfdb.dl_database(db, str(raw_dir), records=missing)
        except Exception as e:
            # A batch failure (e.g. one record 404s) shouldn't lose records that WOULD have
            # downloaded fine -- fall back to one at a time and skip+log only the failures.
            print(f"  batch download raised {e!r}; retrying records individually")
            for r in missing:
                if have(r):
                    continue
                try:
                    wfdb.dl_database(db, str(raw_dir), records=[r])
                except Exception as e2:
                    print(f"  SKIP {db}/{r}: download failed ({e2!r})")
    present = [r for r in records if have(r)]
    skipped = sorted(set(records) - set(present))
    if skipped:
        print(f"  {db}: {len(skipped)} record(s) unavailable and skipped: {skipped}")
    return present


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_record(record_path: Path, db: str = "mitdb"):
    """Filter the whole record, then cut beats. Returns (arrays dict, skip-reason Counter).

    `db` selects the lead-preference list (config.DB_LEAD_PREFERENCE) and whether the record
    needs resampling to config.FS first. mitdb's branch (db="mitdb", the default) is BYTE-FOR-
    BYTE the same code path as before multi-database support was added: it never calls
    resample_record, even as a no-op -- see the fs check below. This is the one guarantee this
    function must never break; it's covered by a regression check that diffs a fresh MIT-BIH
    build against data/processed/test.npz before any multi-database build is trusted.

    Drops beats with an implausible RR gap (config.RR_MIN_S/RR_MAX_S) at build time, which
    keeps the annotation-gap outliers out of train's statistics entirely. This is
    complementary to, not redundant with, the clip in preprocessing.fit_rr_scaler/
    apply_rr_scaler: dropping protects the offline dataset (and prevents a gap beat's own
    sample from ever being learned from), while the scaler's clip protects a future
    streaming beat at inference time, which can't simply be dropped mid-stream, and also
    catches beats whose OWN pre/post RR is plausible but whose pre_rr_ratio/post_rr_ratio is
    still contaminated because a neighboring beat's gap skewed the local-median denominator.
    """
    import wfdb

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    lead_prefs = C.DB_LEAD_PREFERENCE[db]
    lead = next((l for l in lead_prefs if l in rec.sig_name), None)
    if lead is None:
        raise ValueError(f"{record_path}: none of {lead_prefs} in {rec.sig_name}")
    raw_signal = rec.p_signal[:, rec.sig_name.index(lead)]

    if rec.fs != C.FS:
        if db == "mitdb":
            raise ValueError(f"{record_path}: expected fs={C.FS} for mitdb, got {rec.fs}")
        raw_signal, samples = resample_record(raw_signal, ann.sample, rec.fs)
    else:
        samples = np.asarray(ann.sample, dtype=np.int64)

    # Filter the full 30-minute signal once. Filtering happens BEFORE segmentation.
    signal = bandpass(raw_signal)

    symbols = np.asarray(ann.symbol)
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
        pre_rr_s, post_rr_s = rr_all[i, 0], rr_all[i, 1]
        if not (C.RR_MIN_S <= pre_rr_s <= C.RR_MAX_S and C.RR_MIN_S <= post_rr_s <= C.RR_MAX_S):
            # Finite but physiologically implausible: an annotation gap, not a real beat-
            # to-beat interval (see config.RR_MIN_S/RR_MAX_S). Same kind of untrustworthy RR
            # context as the NaN case above, just not NaN, so it gets its own counter.
            skipped["implausible_rr_gap"] += 1
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
                       target: float = 0.2, max_frac: float = 0.5,
                       max_exhaustive: int = 200_000, sample_combos: int = 50_000,
                       seed: int = 0):
    """Pick whole records for validation so each class lands near `target` fraction.

    Minority classes are concentrated in a few records (most F beats sit in one record),
    so a random record pick can leave a class missing from train or val. This searches
    combinations of `sizes` records and rejects any that take more than `max_frac` of a
    class away from training.

    Exhaustive by default (DS1's 22-record pool: 33,649 combinations for sizes=(4,5), fast).
    For a larger candidate pool -- e.g. INCART's 75 records, where C(75,15) alone is in the
    hundreds of billions -- exhaustive enumeration is infeasible. When the combination count
    for a given size exceeds `max_exhaustive`, this falls back to `sample_combos` random
    combinations of that size (seeded, reproducible) instead: an approximate search, not
    exact, but the same scoring rule, and honestly not exhaustive -- don't assume its result
    is provably optimal the way DS1's small-pool result is.
    """
    candidates = sorted(candidates)
    totals = {c: sum(per_record_counts[r][c] for r in candidates) for c in C.CLASSES}
    active = [c for c in C.CLASSES if totals[c] > 0]
    rng = np.random.default_rng(seed)

    def score_combo(combo):
        fracs = {c: sum(per_record_counts[r][c] for r in combo) / totals[c] for c in active}
        if any(f > max_frac for f in fracs.values()):
            return None, None
        score = sum((f - target) ** 2 for f in fracs.values())
        score += sum(1.0 for f in fracs.values() if f == 0)  # strongly avoid empty classes
        return score, fracs

    best, best_score, best_fracs = None, float("inf"), None
    for k in sizes:
        n_combos = math.comb(len(candidates), k)
        if n_combos <= max_exhaustive:
            combos = itertools.combinations(candidates, k)
        else:
            # Random k-subsets, not a slice of the exhaustive enumeration -- avoids the bias
            # of e.g. only ever sampling combinations that start with early candidates.
            combos = (tuple(rng.choice(candidates, size=k, replace=False).tolist())
                      for _ in range(sample_combos))
        for combo in combos:
            score, fracs = score_combo(combo)
            if score is not None and score < best_score:
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


def subsample_n(data: dict, cap: int, seed: int = 0) -> dict:
    """Randomly subsample class-N beats down to `cap` (no-op if already <= cap); every
    S/V/F beat is kept untouched. Used for external training sources only (INCART, SVDB) --
    MIT-BIH's own N beats are never subsampled."""
    y = data["y"]
    n_mask = y == C.CLASS_TO_IDX["N"]
    n_idx = np.where(n_mask)[0]
    if len(n_idx) > cap:
        rng = np.random.default_rng(seed)
        n_idx = rng.choice(n_idx, size=cap, replace=False)
    keep = np.sort(np.concatenate([n_idx, np.where(~n_mask)[0]]))
    return {k: v[keep] for k, v in data.items()}


def build_external_source(raw_dir: Path, records: tuple, db: str, skip_download: bool) -> tuple[dict, dict, Counter, list]:
    """Download (unless skipped) and process every record of one external database.

    Returns (per_record data dict, per_record class_counts dict, combined skipped-beat
    Counter, list of records that failed to process after downloading -- corrupt/unreadable,
    not just missing). Never raises on a single bad record; that's the point.
    """
    if skip_download:
        present = [r for r in records if (raw_dir / f"{r}.hea").exists()]
    else:
        present = ensure_downloaded(raw_dir, records, db=db)

    per_record, counts, skipped_total, failed = {}, {}, Counter(), []
    for r in present:
        try:
            data, skipped = process_record(raw_dir / r, db=db)
        except Exception as e:
            print(f"  SKIP {db}/{r}: failed to process ({e!r})")
            failed.append(r)
            continue
        per_record[r], counts[r] = data, class_counts(data["y"])
        skipped_total.update(skipped)
    return per_record, counts, skipped_total, failed


def print_table(title: str, rows: dict) -> None:
    header = f"{'':>8}" + "".join(f"{c:>8}" for c in C.CLASSES) + f"{'total':>9}"
    print(f"\n{title}\n{header}")
    for name, counts in rows.items():
        total = sum(counts.values())
        print(f"{name:>8}" + "".join(f"{counts[c]:>8}" for c in C.CLASSES) + f"{total:>9}")


# ---------------------------------------------------------------------------
# Phase 2.75: multi-database build (MIT-BIH + INCART + SVDB)
# ---------------------------------------------------------------------------
def build_multidb(args: argparse.Namespace) -> dict:
    """BUILD STAGE ONLY: produces data/processed_multidb/*.npz + metadata_multidb.json.
    Does not train anything and does not decide the SVDB ablation -- that's Stage B.

    Split protocol (locked design, see PR/dispatch notes, not re-litigated here):
      - DS2 (MIT-BIH) stays the frozen canonical test set, exactly as the single-db build
        produces it. This function's FIRST job is to prove that with a byte-for-byte diff
        against the existing data/processed/test.npz -- everything after that gate assumes
        it passed.
      - MIT-BIH DS1-val + INCART-val become ONE pooled validation set (kept as two separate
        files here; Stage B's training loop pools them for its selection criterion and keeps
        each as a separate diagnostic -- see train.py's SELECTION_CLASSES comment for why
        that separation matters).
      - INCART-holdout is carved patient-disjoint from both INCART-val and INCART-train, and
        is not touched by anything else in this function after that carve.
      - SVDB is processed and N-subsampled the same way as INCART, but kept as a SEPARATE
        pool file (not merged into train.npz) so Stage B can ablate with/without it cleanly.
    """
    report: dict = {}

    # --- Step 1: MIT-BIH, unchanged pipeline (db="mitdb" never resamples) ---
    print("=== MIT-BIH ===")
    mitbih_raw = Path("data/raw/mitdb")
    mitbih_records = list(C.DS1) + list(C.DS2)
    if not args.skip_download:
        ensure_downloaded(mitbih_raw, mitbih_records, db="mitdb")
    mit_per_record, mit_counts = {}, {}
    for r in mitbih_records:
        data, _skipped = process_record(mitbih_raw / r, db="mitdb")
        mit_per_record[r], mit_counts[r] = data, class_counts(data["y"])
    print_table("MIT-BIH beats per record", mit_counts)

    mit_val, mit_val_fracs = choose_val_records(mit_counts, C.DS1)
    mit_train = [r for r in C.DS1 if r not in mit_val]
    mit_test = list(C.DS2)
    test_split = concat_split(mit_per_record, mit_test)
    val_mitbih_split = concat_split(mit_per_record, mit_val)
    mit_train_split = concat_split(mit_per_record, mit_train)

    # --- Step 2: DS2 regression gate -- must pass before anything below is trusted ---
    print("\n=== DS2 regression check vs data/processed/test.npz ===")
    existing_test_path = Path("data/processed/test.npz")
    regression = {"checked": False, "identical": None, "detail": None}
    if existing_test_path.exists():
        existing = np.load(existing_test_path)
        detail = {k: bool(np.array_equal(existing[k], test_split[k]))
                 for k in ("X", "rr", "y", "sample", "symbol", "record")}
        identical = all(detail.values())
        regression = {"checked": True, "identical": identical, "detail": detail}
        print(f"  identical: {identical}  (per-array: {detail})")
        if not identical:
            raise RuntimeError(
                f"DS2 regression check FAILED: {detail}. Aborting before building anything "
                "on top of it -- the multi-db refactor changed MIT-BIH's own output."
            )
    else:
        print(f"  WARNING: {existing_test_path} not found -- cannot regression-check DS2 "
              "(run the single-db build first for this gate to mean anything).")
    report["ds2_regression"] = regression

    # --- Step 3: INCART ---
    print("\n=== INCART ===")
    incart_raw = Path("data/raw/incartdb")
    incart_per_record, incart_counts, incart_skipped, incart_failed = build_external_source(
        incart_raw, C.INCART_RECORDS, db="incartdb", skip_download=args.skip_download)
    incart_present = sorted(incart_per_record)
    print_table("INCART beats per record", incart_counts)
    if incart_failed:
        print(f"  INCART records unavailable/failed: {incart_failed}")

    incart_val, incart_val_fracs = choose_val_records(
        incart_counts, incart_present, sizes=(14, 15, 16), target=0.2, max_frac=0.5, seed=args.seed)
    remaining = [r for r in incart_present if r not in incart_val]
    incart_holdout, incart_holdout_fracs = choose_val_records(
        incart_counts, remaining, sizes=(14, 15, 16), target=0.2, max_frac=0.5, seed=args.seed + 1)
    incart_train_records = [r for r in remaining if r not in incart_holdout]

    assert not (set(incart_val) & set(incart_holdout)), "INCART val/holdout must be disjoint"
    assert not (set(incart_val) & set(incart_train_records)), "INCART val/train must be disjoint"
    assert not (set(incart_holdout) & set(incart_train_records)), "INCART holdout/train must be disjoint"

    val_incart_split = concat_split(incart_per_record, incart_val)
    holdout_incart_split = concat_split(incart_per_record, incart_holdout)
    incart_train_split = concat_split(incart_per_record, incart_train_records)

    incart_val_f = sum(incart_counts[r]["F"] for r in incart_val)
    incart_holdout_f = sum(incart_counts[r]["F"] for r in incart_holdout)
    f_warnings = []
    if incart_val_f == 0:
        f_warnings.append("INCART-val has ZERO F beats -- F is too thin/concentrated in this "
                          "source to guarantee a non-degenerate val slice.")
    if incart_holdout_f == 0:
        f_warnings.append("INCART-holdout has ZERO F beats -- same reason.")
    for w in f_warnings:
        print(f"  WARNING: {w}")

    # --- Step 4: SVDB (train-pool only, kept separate for the later ablation) ---
    print("\n=== SVDB ===")
    svdb_raw = Path("data/raw/svdb")
    svdb_per_record, svdb_counts, svdb_skipped, svdb_failed = build_external_source(
        svdb_raw, C.SVDB_RECORDS, db="svdb", skip_download=args.skip_download)
    print_table("SVDB beats per record", svdb_counts)
    if svdb_failed:
        print(f"  SVDB records unavailable/failed: {svdb_failed}")
    svdb_split = concat_split(svdb_per_record, sorted(svdb_per_record))

    # --- Step 5: N-subsample cap, PER CLASS against MIT-BIH DS1-train's own class ratios --
    # not a single combined S+V+F bucket. A combined bucket lets an abundant class (V, which
    # both external sources contribute heavily) mask a scarce one (S) staying just as scarce
    # relative to N -- exactly what happened in an earlier version of this cap: it computed
    # to a no-op because INCART's V volume inflated the combined-minority denominator, while
    # INCART-train's own N:S ratio (95051:1234 =~ 77:1) was actually WORSE than MIT-BIH's own
    # (49:1), so the pooled train set's N:S ended up worse than before this phase started --
    # the opposite of the point of adding S-bearing external data. Capping per class fixes
    # that: for each external source, N is capped so neither the resulting N:S nor N:V ratio
    # exceeds MIT-BIH DS1-train's own ratio for that class (mit_ratio[c] = mit_train_N /
    # mit_train_count[c]), computed against "MIT-BIH + this one source" so INCART's cap and
    # SVDB's cap don't depend on each other -- each pool stays independently usable for
    # Stage B regardless of which other pools it's combined with.
    #
    # F is DELIBERATELY EXCLUDED from the binding constraint, argued, not left implicit: F is
    # already confirmed unlearnable under this architecture regardless of class weighting (see
    # backend-engineer-expert's Phase 2.5 report -- 97.9% of MIT-BIH's own train F beats sit in
    # one record, F's CrossEntropyLoss weight was already pushed to its practical limit and it
    # still didn't generalize). Binding the N cap to F's ratio as well would force discarding
    # the large majority of otherwise-useful external N/S/V beats to protect a ratio for a
    # class that empirically does not respond to ratio protection -- a bad trade. F's absolute
    # count still grows from every source (kept in full, never subsampled), it just doesn't
    # dictate how much N the pool is allowed to keep.
    mit_train_counts = {c: sum(mit_counts[r][c] for r in mit_train) for c in C.CLASSES}
    mit_ratio = {c: (mit_train_counts["N"] / mit_train_counts[c]) if mit_train_counts[c] else None
                for c in ("S", "V", "F")}

    def n_cap_against_mitbih(source_counts: dict) -> int:
        """Cap on (mitbih_N + source_N) so neither N:S nor N:V exceeds MIT-BIH DS1-train's own
        ratio, given this source's S/V beats added to MIT-BIH's. Returns the ALLOWANCE for the
        source's own N (not the pooled total) -- i.e. already has mit_train_counts['N']
        subtracted, clipped at 0."""
        pooled = {c: mit_train_counts[c] + source_counts.get(c, 0) for c in ("S", "V")}
        bounds = [mit_ratio[c] * pooled[c] for c in ("S", "V") if mit_ratio[c] is not None]
        pooled_n_cap = min(bounds) if bounds else float("inf")
        return max(0, round(pooled_n_cap) - mit_train_counts["N"])

    incart_source_counts = class_counts(incart_train_split["y"])
    incart_n_cap = n_cap_against_mitbih(incart_source_counts)
    incart_train_split = subsample_n(incart_train_split, incart_n_cap, seed=args.seed)

    svdb_source_counts = class_counts(svdb_split["y"])
    svdb_n_cap = n_cap_against_mitbih(svdb_source_counts)
    svdb_split = subsample_n(svdb_split, svdb_n_cap, seed=args.seed)

    # --- Step 6: pooled training set = MIT-BIH DS1-train UNION INCART-train (SVDB stays a
    # separate file; it is not merged in here -- that decision belongs to the ablation). ---
    train_split = {k: np.concatenate([mit_train_split[k], incart_train_split[k]])
                  for k in mit_train_split}

    out_dir = args.out_dir_multidb
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "test": test_split,                    # DS2, MIT-BIH only, byte-identical to today
        "val_mitbih": val_mitbih_split,         # MIT-BIH DS1-val
        "val_incart": val_incart_split,         # INCART-val
        "holdout_incart": holdout_incart_split, # INCART-holdout, touched only at final eval
        "train": train_split,                   # MIT-BIH DS1-train + INCART-train (N-capped)
        "svdb_train_pool": svdb_split,          # separate; Stage B ablates with/without this
    }
    for name, split in files.items():
        np.savez_compressed(out_dir / f"{name}.npz", **split)

    split_counts = {name: class_counts(split["y"]) for name, split in files.items()}
    print_table("\nMulti-db beats per output file", split_counts)

    # Both training arms for the (not-yet-run) SVDB ablation, reported here so Stage B doesn't
    # have to re-derive them: "without" = train.npz alone, "with" = train.npz + svdb_train_pool.
    with_svdb_counts = {c: split_counts["train"][c] + split_counts["svdb_train_pool"][c]
                        for c in C.CLASSES}

    # Explicit patient/record-disjointness check across ALL FIVE output files, not just the
    # three INCART ones -- MIT-BIH and INCART use disjoint naming schemes (numeric vs "I##")
    # so collision is not possible by construction, but this checks it rather than assuming.
    record_sets = {name: set(np.unique(split["record"]).tolist()) for name, split in files.items()
                   if "record" in split}
    disjointness = {}
    names = list(record_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(record_sets[a] & record_sets[b])
            disjointness[f"{a} vs {b}"] = overlap  # empty list == disjoint

    report.update({
        "n_cap_method": "per-class (S and V) against MIT-BIH DS1-train's own ratios, "
                        "independently per external source; F deliberately excluded from the "
                        "binding constraint -- see code comment in build_multidb for why",
        "mit_train_ratio_N_to_S": mit_ratio["S"], "mit_train_ratio_N_to_V": mit_ratio["V"],
        "mit_train_ratio_N_to_F": mit_ratio["F"],
        "incart_n_cap": incart_n_cap, "incart_source_counts_pre_subsample": incart_source_counts,
        "svdb_n_cap": svdb_n_cap, "svdb_source_counts_pre_subsample": svdb_source_counts,
        "incart_val_records": incart_val, "incart_holdout_records": incart_holdout,
        "incart_train_records": incart_train_records,
        "incart_val_fracs": incart_val_fracs, "incart_holdout_fracs": incart_holdout_fracs,
        "incart_val_f_beats": incart_val_f, "incart_holdout_f_beats": incart_holdout_f,
        "incart_f_warnings": f_warnings,
        "incart_records_unavailable": incart_failed,
        "svdb_records_unavailable": svdb_failed,
        "svdb_records_requested": list(C.SVDB_RECORDS),
        "split_counts": split_counts,
        "with_svdb_train_counts": with_svdb_counts,
        "without_svdb_train_counts": split_counts["train"],
        "record_disjointness_check": disjointness,
        "mit_val_records": mit_val, "mit_train_records": mit_train,
    })
    (out_dir / "metadata_multidb.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved multi-db build to {out_dir.resolve()}")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/mitdb"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--val-records", nargs="+", default=None,
                    help="DS1 records to use for validation (default: chosen automatically)")
    ap.add_argument("--multidb", action="store_true",
                    help="Phase 2.75 BUILD STAGE ONLY: build the MIT-BIH+INCART(+SVDB pool) "
                         "dataset in data/processed_multidb/ instead of the default MIT-BIH-"
                         "only build. Does not train anything.")
    ap.add_argument("--out-dir-multidb", type=Path, default=Path("data/processed_multidb"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.multidb:
        build_multidb(args)
        return

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
            f"Beats with pre_rr_s/post_rr_s outside [{C.RR_MIN_S}, {C.RR_MAX_S}] seconds are "
            "dropped (skipped_beats.implausible_rr_gap): these are MIT-BIH annotation gaps "
            "(dropped/unreadable annotations), not real asystole, and were inflating RR "
            "normalization statistics before this fix.",
        ],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nValidation records: {val}")
    print(f"Skipped beats: {dict(skipped_total)}")
    print(f"Saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
