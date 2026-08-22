"""Loading the processed splits with per-channel z-scoring.

Normalisation statistics are always computed on the TRAIN split and reused
for val/test; each run stores the statistics it used in ``norm_stats.npz``
so evaluation is exact."""
from pathlib import Path

import numpy as np
import torch

from .paths import PROC


def load_split(name, stats=None, proc=PROC):
    """Return (X, y, stats). X: (N,128,9) float32 z-scored; y: int64 in 0..5."""
    d = np.load(Path(proc) / f"{name}.npz")
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64) - 1
    if stats is None:
        flat = X.reshape(-1, X.shape[-1])
        stats = flat.mean(0), flat.std(0) + 1e-8
    X = (X - stats[0]) / stats[1]
    return torch.from_numpy(X), torch.from_numpy(y), stats


def load_split_raw(name, proc=PROC):
    """Unnormalised arrays plus subject ids (for analyses)."""
    d = np.load(Path(proc) / f"{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64) - 1, d["subject"].astype(np.int64)


def load_norm_stats(run_dir):
    s = np.load(Path(run_dir) / "norm_stats.npz")
    return s["mean"], s["std"]


def save_norm_stats(run_dir, stats):
    np.savez(Path(run_dir) / "norm_stats.npz", mean=stats[0], std=stats[1])
