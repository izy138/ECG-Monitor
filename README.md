# ECG Monitor — Heartbeat Classification System

Classifies individual heartbeats from MIT-BIH ECG recordings (Normal, Supraventricular,
Ventricular, Fusion) and streams results to a live dashboard.

**Status:** Phase 1 (data pipeline), Phase 2 (training) and Phase 2.5 (RR-feature and
class-weighting fixes) built. Phases 3–5 to come.

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

## Results (inter-patient, DS2 held out)

| class | precision | recall | F1 | support |
|-------|-----------|--------|-----|---------|
| N     | 0.960 | 0.982 | 0.971 | 44159 |
| S     | 0.305 | 0.168 | 0.216 | 1809 |
| V     | 0.873 | 0.920 | 0.896 | 3219 |
| F     | 0.000 | 0.000 | 0.000 | 388 |

Test accuracy 0.940, macro-F1 0.521. Validation accuracy 0.976, macro-F1 0.653 (4-class;
0.856 on the N/S/V selection criterion actually used to pick this checkpoint — see below).

S's headline recall (0.168) is dragged down by one patient: record 232 holds 1354 of test's
1809 S beats (74.8%) at 0.013 recall, because that patient's S beats arrive at a normal-ish
interval (`pre_rr_ratio` median 0.99) while their N beats are unusually late (`pre_rr_ratio`
median 2.51) — the "S arrives early relative to N" relationship the model learns everywhere
else is inverted in this one patient. **S recall excluding record 232 is 0.626** (455 beats).
Both figures are in `models/metrics.json` (`splits.test.per_record_recall`,
`splits.test.s_recall_excluding_record_232`) so this isn't just a conversational aside.

These are true inter-patient numbers. Intra-patient splits (beats from the same patient in
both train and test) routinely report 95–98% accuracy on this dataset; those figures are not
comparable to these and are not a target.

**Class F is not learnable under this split, and is reported as a known limitation rather
than a solved class.** 372 of the 380 fusion beats in training come from a single record
(208), and 362 of the 388 in test come from a different single record (213). The validation
fusion beats come from four further records, none of them 208. So the model is asked to
learn "fusion" from essentially one patient and recognize it in another — and fusion beats
are by definition morphologically intermediate between N and V, so there is little
patient-independent shape to extract. See `--class-weighting` below for why we stopped
trying to force it.

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
- **RR features are clipped to physiological bounds before normalization** (`RR_MIN_S`,
  `RR_MAX_S`, `RR_RATIO_MAX` in `config.py`). MIT-BIH annotation files contain gaps of up to
  100 seconds between consecutive beat annotations — dropped annotations, not asystole.
  Six such beats were enough to inflate the training RR standard deviation to 0.549 (the
  99.9th percentile is 1.89), compressing the real rhythm distribution into an interquartile
  range of ±0.07σ and making the RR features invisible to the model. Beats whose own pre/post
  RR is implausible are dropped at build time (`skipped_beats.implausible_rr_gap`, 80 beats);
  the clip additionally protects streaming inference, where a beat cannot be dropped
  mid-stream, and catches beats whose local-median denominator was skewed by a neighbor's gap.
  The fitted scaler is stored in the checkpoint, so serving cannot drift from training.
- **`--class-weighting` defaults to `sqrt_inverse`, not plain inverse-frequency.** Inverse
  weighting gives F a ~94× true-label loss advantage over N, which the optimizer exploits by
  sacrificing true-N beats wholesale: on validation it produced 642 N→F false positives to
  catch 7 of 34 true fusion beats, dropping N recall from 0.997 to 0.930. `sqrt_inverse`
  compresses that ratio to ~9.7×. Pass `--class-weighting inverse` to reproduce the old
  behavior.
- **Checkpoint selection uses val macro-F1 over N/S/V only** (`SELECTION_CLASSES` in
  `train.py`), not the full 4-class average. F is excluded because it's confirmed unlearnable
  from this data (see above), not because excluding it makes selection less noisy — it
  doesn't: the N/S/V-only band's coefficient of variation (~0.041, epochs 3–16) is essentially
  identical to the 4-class band's (~0.041). Selection is noisy either way; the source is
  minority-class val counts in general (S has only 213 val beats), not F specifically. F stays
  a trained output class and stays in every reported number regardless of this flag.
- **`--weight-decay` defaults to `1e-4`, a real but marginal fix for that selection noise, not
  a solved problem.** train_loss falls steeply (0.458→0.053 over 16 epochs) while val_loss
  stays flat/noisy from ~epoch 4 on — classic overfitting. Adam weight decay was compared
  against BeatCNN's alternative (raising its single head-layer `dropout`, which would leave
  the conv feature extractor — most of the model's capacity — untouched): 1e-4 lowers the
  N/S/V selection-F1 band's std from 0.033 to 0.025 (CoV 0.041→0.031) and final-epoch val_loss
  from 0.612 to 0.603, but the selected epoch and its metric barely move (0.8585→0.8565, same
  epoch). 1e-3 is worse, not just unhelpful — it reintroduces N→F confusion (4→164 beats) and
  drops S's F1 (0.73→0.60) without closing the loss gap either.
- Known caveat: records 201 (DS1) and 202 (DS2) come from the same patient.
