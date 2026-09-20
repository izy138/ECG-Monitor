"""Phase 2: train a beat classifier on the processed dataset.

    python -m ecg.train
    python -m ecg.train --epochs 30 --batch-size 256
    python -m ecg.train --evaluate-only --checkpoint models/best.pt

Early-stops on validation macro-F1. The test set is evaluated once at the end (or via
--evaluate-only) and must not be used for model selection.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from . import config as C  # noqa: E402
from .model import BeatCNN  # noqa: E402
from .preprocessing import apply_rr_scaler, fit_rr_scaler  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def shift_window(window: np.ndarray, shift: int) -> np.ndarray:
    """Simulate R-peak misalignment by shifting the beat window ± a few samples."""
    out = np.zeros_like(window)
    if shift > 0:
        out[shift:] = window[:-shift]
    elif shift < 0:
        out[:shift] = window[-shift:]
    else:
        out[:] = window
    return out


class BeatDataset(Dataset):
    def __init__(self, X: np.ndarray, rr: np.ndarray, y: np.ndarray,
                 rr_scaler: dict, jitter: int = 0, seed: int = 0):
        self.X = X
        self.rr = apply_rr_scaler(rr, rr_scaler)
        self.y = y.astype(np.int64)
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.jitter > 0:
            shift = int(self.rng.integers(-self.jitter, self.jitter + 1))
            if shift != 0:
                x = shift_window(x, shift)
        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.from_numpy(self.rr[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["rr"], data["y"]


def class_weights(y: np.ndarray, scheme: str = "inverse") -> torch.Tensor:
    """Per-class weights for CrossEntropyLoss's `weight=` argument.

    "inverse" is plain inverse-frequency (the original scheme): weight = total / (n_classes *
    count). With CrossEntropyLoss, that weight is indexed by each sample's TRUE label, and
    default reduction='mean' divides by the sum of those per-sample weights -- so a false
    negative on a rare class costs (weight[rare] / weight[common]) times as much as an
    equally-confident false positive on that rare class. For this dataset's train counts
    (N 35879, S 730, V 3150, F 380) that ratio is 26.4 / 0.28 = ~94x in F's favor, which is
    large enough to make the optimizer trade many cheap true-N misses for a few expensive
    true-F hits (see backend-engineer-expert's diagnosis for confirming numbers).

    "sqrt_inverse" tempers that: weight = sqrt(inverse-frequency weight). It keeps the same
    ordering (rarer classes still weighted more) but compresses the ratio between the most
    and least frequent classes from ~94x to ~sqrt(94) ~= 9.7x, so a rare class still gets
    priority without being worth sacrificing a large share of the majority class for.
    """
    counts = np.bincount(y, minlength=len(C.CLASSES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(C.CLASSES) * counts)
    if scheme == "sqrt_inverse":
        weights = np.sqrt(weights)
    elif scheme != "inverse":
        raise ValueError(f"Unknown class-weighting scheme: {scheme!r}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: BeatCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for x, rr, y in loader:
        x, rr = x.to(device), rr.to(device)
        logits = model(x, rr)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def run_epoch(model: BeatCNN, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n = 0
    for x, rr, y in loader:
        x, rr, y = x.to(device), rr.to(device), y.to(device)
        logits = model(x, rr)
        loss = criterion(logits, y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=C.CLASSES, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(C.CLASSES))))
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=C.CLASSES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    header = "     " + "".join(f"{c:>6}" for c in C.CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{C.CLASSES[i]:>4} " + "".join(f"{v:>6}" for v in row))
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": {c: report[c] for c in C.CLASSES},
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args: argparse.Namespace) -> Path:
    device = pick_device()
    print(f"Device: {device}")

    X_train, rr_train, y_train = load_split(args.data_dir / "train.npz")
    X_val, rr_val, y_val = load_split(args.data_dir / "val.npz")
    rr_scaler = fit_rr_scaler(rr_train)  # TRAIN split only

    train_ds = BeatDataset(X_train, rr_train, y_train, rr_scaler,
                           jitter=args.jitter, seed=args.seed)
    val_ds = BeatDataset(X_val, rr_val, y_val, rr_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BeatCNN().to(device)
    weights = class_weights(y_train, scheme=args.class_weighting).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "best.pt"
    history = []
    best_f1 = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        y_pred, y_true = predict(model, val_loader, device)
        val_f1 = macro_f1(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale = 0
            torch.save({
                "model_state": model.state_dict(),
                "rr_scaler": rr_scaler,
                "val_macro_f1": val_f1,
                "epoch": epoch,
                "classes": list(C.CLASSES),
            }, ckpt_path)
            print(f"  -> saved checkpoint (val_macro_f1={val_f1:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val F1 improvement for {args.patience} epochs)")
                break

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest validation macro-F1: {best_f1:.4f}")
    return ckpt_path


def evaluate(args: argparse.Namespace, ckpt_path: Path) -> dict:
    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = BeatCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    if "rr_scaler" not in ckpt:
        raise KeyError(
            f"{ckpt_path} has no 'rr_scaler' entry -- it was saved by an older version of "
            "train.py (separate rr_mean/rr_std, or an even earlier format). Retrain with the "
            "current code (python -m ecg.train) to produce a compatible checkpoint."
        )
    rr_scaler = ckpt["rr_scaler"]

    results = {}
    for split in ("val", "test"):
        path = args.data_dir / f"{split}.npz"
        if not path.exists():
            continue
        X, rr, y = load_split(path)
        loader = DataLoader(
            BeatDataset(X, rr, y, rr_scaler),
            batch_size=args.batch_size, shuffle=False,
        )
        y_pred, y_true = predict(model, loader, device)
        results[split] = evaluate_split(split, y_true, y_pred)
        save_confusion_matrix(
            np.array(results[split]["confusion_matrix"]),
            args.out_dir / f"confusion_{split}.png",
            f"{split} confusion matrix",
        )

    report = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_macro_f1": ckpt.get("val_macro_f1"),
        "splits": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved metrics to {args.out_dir / 'metrics.json'}")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--jitter", type=int, default=5, help="± samples of R-peak shift augmentation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weighting", choices=["inverse", "sqrt_inverse"], default="sqrt_inverse",
                    help="CrossEntropyLoss class weighting scheme (see class_weights() docstring). "
                         "sqrt_inverse is the default: plain inverse-frequency weighting gives F "
                         "a ~94x true-label loss advantage over N, which the optimizer exploits by "
                         "sacrificing hundreds of true-N beats to catch a handful of true-F beats "
                         "(val: 642 N->F false positives for 7/34 true F caught). sqrt_inverse "
                         "compresses that ratio to ~9.7x, eliminating the N regression at the cost "
                         "of F recall reverting to near-zero -- an explicit, evidence-based trade "
                         "(see backend-engineer-expert's report), not a default anyone should flip "
                         "back without re-running that comparison on val.")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.evaluate_only:
        ckpt = args.checkpoint or args.out_dir / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        evaluate(args, ckpt)
        return

    ckpt = train(args)
    evaluate(args, ckpt)


if __name__ == "__main__":
    main()
