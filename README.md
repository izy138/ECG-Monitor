# ECG Monitor — Heartbeat Classification System

Classifies individual heartbeats from MIT-BIH ECG recordings (Normal, Supraventricular,
Ventricular, Fusion) and streams results to a live dashboard.

**Status:** Phase 1 (data pipeline) and Phase 2 (training) built. Phases 3–5 to come.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # 11 tests, uses synthetic WFDB records, no download
```

## Phase 1 — build the dataset

```bash
python -m ecg.build_dataset             # downloads MIT-BIH (~100 MB) on first run
python -m ecg.inspect_dataset           # writes data/processed/inspect_train.png
```

Outputs in `data/processed/`: `train.npz`, `val.npz`, `test.npz`, `metadata.json`.

## Phase 2 — train the classifier

```bash
python -m ecg.train                    # trains with early stopping on val macro-F1
python -m ecg.train --epochs 30 --batch-size 256
python -m ecg.train --evaluate-only   # re-run metrics from models/best.pt
```

Outputs in `models/`: `best.pt`, `history.json`, `metrics.json`, `confusion_val.png`, `confusion_test.png`.

The CNN takes a 200-sample beat plus 4 RR-interval features. Training uses class-weighted
cross-entropy and ±5-sample R-peak jitter. The test set is evaluated once at the end — never
used for early stopping.

## Layout

```
ecg/
  config.py          constants shared by training AND inference (window, filter, labels, splits)
  preprocessing.py   bandpass, windowing, z-score, RR-interval features
  build_dataset.py   Phase 1 CLI
  inspect_dataset.py sanity-check plots
  model.py           1D CNN + RR head
  train.py           Phase 2 CLI
tests/
```

## Methodology decisions

- **Inter-patient evaluation** (de Chazal et al., 2004): train on DS1, test on DS2. Paced
  records (102, 104, 107, 217) excluded, so the problem is 4-class (N, S, V, F).
- **Validation is carved from DS1 by whole record**, chosen automatically so each class
  lands near 20%. DS2 is never used for early stopping or model selection.
- **Filtering happens on the full record before segmentation** (Butterworth 0.5–45 Hz,
  second-order sections, zero-phase).
- **RR-interval features** (pre/post RR and their ratio to the local median rhythm) are
  saved alongside each waveform, because supraventricular beats are identified mainly by
  timing, not shape.
- Known caveat: records 201 (DS1) and 202 (DS2) come from the same patient.
