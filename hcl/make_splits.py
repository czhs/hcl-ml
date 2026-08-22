#!/usr/bin/env python3
"""Build subject-wise train/val/test splits from the UCI HAR Inertial Signals.

The official subject-wise train/test split is preserved untouched. Validation
is carved out of the TRAINING subjects only (4 subjects, chosen with SEED=42),
so val measures cross-subject generalisation exactly like test does and the
test set is never used for model selection.

Outputs (under --out):
  train.npz / val.npz / test.npz   X (N,128,9) float32, y (N,) int64 1-6, subject (N,) int64
  splits.json                      subjects, counts, per-channel statistics
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROC, RAW

SEED = 42
N_VAL_SUBJECTS = 4

# Canonical channel order used everywhere downstream.
SIGNALS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

ACTIVITIES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}


def load_official_split(root: Path, split: str):
    sig_dir = root / split / "Inertial Signals"
    chans = [
        pd.read_csv(sig_dir / f"{s}_{split}.txt", sep=r"\s+", header=None).to_numpy(np.float32)
        for s in SIGNALS
    ]
    X = np.stack(chans, axis=-1)  # (N, 128, 9)
    y = pd.read_csv(root / split / f"y_{split}.txt", header=None).to_numpy(np.int64).ravel()
    subj = pd.read_csv(root / split / f"subject_{split}.txt", header=None).to_numpy(np.int64).ravel()
    assert X.shape[1:] == (128, 9), X.shape
    assert len(X) == len(y) == len(subj)
    return X, y, subj


def class_counts(y):
    return {ACTIVITIES[c]: int((y == c).sum()) for c in sorted(ACTIVITIES)}


def channel_stats(X):
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    return {
        s: {"mean": float(flat[:, i].mean()), "std": float(flat[:, i].std()),
            "min": float(flat[:, i].min()), "max": float(flat[:, i].max())}
        for i, s in enumerate(SIGNALS)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RAW), help="unpacked 'UCI HAR Dataset' directory")
    ap.add_argument("--out", default=str(PROC))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
    if not (root / "train" / "Inertial Signals").exists():
        raise SystemExit(f"dataset not found at {root} — run setup.sh or set HCL_DATA")
    out.mkdir(parents=True, exist_ok=True)

    X_tr_full, y_tr_full, s_tr_full = load_official_split(root, "train")
    X_te, y_te, s_te = load_official_split(root, "test")

    train_subjects_full = sorted(np.unique(s_tr_full).tolist())
    test_subjects = sorted(np.unique(s_te).tolist())
    assert not set(train_subjects_full) & set(test_subjects), "train/test subjects overlap!"

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(train_subjects_full)
    val_subjects = sorted(int(s) for s in perm[:N_VAL_SUBJECTS])
    train_subjects = sorted(set(train_subjects_full) - set(val_subjects))

    val_mask = np.isin(s_tr_full, val_subjects)
    splits = {
        "train": (X_tr_full[~val_mask], y_tr_full[~val_mask], s_tr_full[~val_mask]),
        "val": (X_tr_full[val_mask], y_tr_full[val_mask], s_tr_full[val_mask]),
        "test": (X_te, y_te, s_te),
    }

    meta = {
        "seed": args.seed,
        "signals": SIGNALS,
        "activities": {str(k): v for k, v in ACTIVITIES.items()},
        "window": {"length": 128, "rate_hz": 50.0, "seconds": 2.56, "overlap": 0.5},
        "subjects": {"train": train_subjects, "val": val_subjects, "test": test_subjects},
        "splits": {}, "per_subject": {},
    }
    for name, (X, y, s) in splits.items():
        np.savez_compressed(out / f"{name}.npz", X=X, y=y, subject=s)
        meta["splits"][name] = {
            "n_windows": int(len(X)),
            "n_subjects": int(len(np.unique(s))),
            "class_counts": class_counts(y),
            "channel_stats": channel_stats(X),
        }
        for subj in np.unique(s):
            m = s == subj
            meta["per_subject"][str(int(subj))] = {
                "split": name, "n_windows": int(m.sum()), "class_counts": class_counts(y[m]),
            }
        print(f"{name:5s}: {len(X):5d} windows, {len(np.unique(s)):2d} subjects, "
              f"classes {list(class_counts(y).values())}")

    total = sum(meta["splits"][k]["n_windows"] for k in meta["splits"])
    meta["fractions"] = {k: round(meta["splits"][k]["n_windows"] / total, 4) for k in meta["splits"]}
    print("fractions:", meta["fractions"])
    print("val subjects:", val_subjects)
    (out / "splits.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / 'splits.json'}")


if __name__ == "__main__":
    main()
