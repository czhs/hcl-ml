#!/usr/bin/env python3
"""Figures from result JSON (matplotlib): confusion matrices, validation curves
and the K-sweep. Writes PNGs to <results>/figures/.

  python -m hcl.plots --results reference   # regenerates the figures used in the note
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import CLASSES, RESULTS

SHORT = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]


def confusions(results, figdir):
    p = results / "test_results.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    keys = [k for k in ("direct", "simplex", "shuffled") if k in d["conditions"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        cm = np.array(d["conditions"][k]["confusion"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
        ax.set_xticks(range(6), SHORT, fontsize=7)
        ax.set_yticks(range(6), SHORT, fontsize=7)
        ax.set_title(f"{k}  macro-F1 {d['conditions'][k]['macro_f1']:.3f}", fontsize=9)
        ax.set_xlabel("predicted")
    np.atleast_1d(axes)[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(figdir / "confusions.png", dpi=150)
    print("wrote", figdir / "confusions.png")


def curves(results, figdir, seed):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    found = False
    for variant in ("direct", "simplex", "free", "bottleneck"):
        for base in (results / "checkpoints", results):
            m = base / f"{variant}_s{seed}" / "metrics.jsonl"
            if m.exists():
                recs = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
                ax.plot([r["val_macro_f1"] for r in recs], label=variant)
                found = True
                break
    if not found:
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-F1")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"validation curves, seed {seed}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "val_curves.png", dpi=150)
    print("wrote", figdir / "val_curves.png")


def ksweep(results, figdir):
    p = results / "all_runs.json"
    if not p.exists():
        return
    runs = json.loads(p.read_text())
    Ks, vals = [], []
    for K in (1, 2, 4, 8, 12, 16, 32, 64):
        r = runs.get(f"simplex-geometric{K}_s0")
        if r:
            Ks.append(K)
            vals.append(r["best_val_macro_f1"])
    if "simplex-geometric_s0" in runs:
        Ks.append(6)
        vals.append(runs["simplex-geometric_s0"]["best_val_macro_f1"])
    if not Ks:
        return
    order = np.argsort(Ks)
    Ks, vals = np.array(Ks)[order], np.array(vals)[order]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(Ks, vals, "o-", label="frozen max-separated corners")
    if "simplex-geometric1-gate_s0" in runs:
        ax.scatter([1], [runs["simplex-geometric1-gate_s0"]["best_val_macro_f1"]], marker="s", color="C3",
                   label="K=1 with sigmoid gate (magnitude code)", zorder=3)
    if "simplex_s0" in runs:
        ax.axhline(runs["simplex_s0"]["best_val_macro_f1"], ls="--", color="gray", label="balanced anchors, K=6")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks, [str(k) for k in Ks])
    ax.set_xlabel("number of simplex corners K")
    ax.set_ylabel("best val macro-F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "ksweep.png", dpi=150)
    print("wrote", figdir / "ksweep.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    figdir = results / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    confusions(results, figdir)
    curves(results, figdir, args.seed)
    ksweep(results, figdir)


if __name__ == "__main__":
    main()
