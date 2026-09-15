"""Eyeball the processed beats before training anything.

    python -m ecg.inspect_dataset --split train

Saves a PNG with, for each class: 20 random beats, the mean waveform with a ±1 std band,
and the distribution of pre-RR ratio (early beats sit left of 1.0).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    d = np.load(args.data_dir / f"{args.split}.npz")
    X, y, rr = d["X"], d["y"], d["rr"]
    t_ms = (np.arange(C.WINDOW_LEN) - C.WINDOW_BEFORE) / C.FS * 1000
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(3, len(C.CLASSES), figsize=(4 * len(C.CLASSES), 9), squeeze=False)
    for j, cls in enumerate(C.CLASSES):
        idx = np.flatnonzero(y == j)
        title = f"{cls} — {C.CLASS_NAMES[cls]} (n={len(idx)})"
        axes[0, j].set_title(title, fontsize=10)
        if len(idx) == 0:
            continue
        for i in rng.choice(idx, size=min(20, len(idx)), replace=False):
            axes[0, j].plot(t_ms, X[i], lw=0.6, alpha=0.5)
        mean, std = X[idx].mean(0), X[idx].std(0)
        axes[1, j].plot(t_ms, mean, color="black")
        axes[1, j].fill_between(t_ms, mean - std, mean + std, alpha=0.25)
        axes[2, j].hist(np.clip(rr[idx, 2], 0, 2.5), bins=50)
        axes[2, j].axvline(1.0, color="red", lw=0.8)
    axes[0, 0].set_ylabel("random beats (z-score)")
    axes[1, 0].set_ylabel("mean ± 1 std")
    axes[2, 0].set_ylabel("count")
    for ax in axes[1]:
        ax.set_xlabel("ms from R-peak")
    for ax in axes[2]:
        ax.set_xlabel("pre-RR / local median RR")
    fig.tight_layout()

    out = args.out or args.data_dir / f"inspect_{args.split}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
