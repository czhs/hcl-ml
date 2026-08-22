#!/usr/bin/env python3
"""Aggregate multi-seed runs: test-evaluate every (variant, seed) checkpoint
and report mean ± std (ddof=1) of val/test macro-F1 per condition.

The shuffled control for seed s uses shuffle seed 123+s, so the permutation
varies with the training seed. Writes <results>/seed_summary.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from . import eval_test as ET
from . import model as M
from .data import load_split
from .paths import RESULTS


def available_seeds(results, variants):
    """Seeds that have a finished checkpoint for EVERY requested variant."""
    per_variant = [
        {int(p.name.split("_s")[-1]) for p in results.glob(f"{v}_s*") if (p / "best.pt").exists()}
        for v in variants
    ]
    return sorted(set.intersection(*per_variant)) if per_variant else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--variants", nargs="+", default=["direct", "simplex", "free", "bottleneck"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: auto-detect")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    seeds = args.seeds or available_seeds(results, args.variants)
    if not seeds:
        raise SystemExit("no seeds with checkpoints for every variant")
    print("seeds:", seeds)

    out = {"seeds": seeds, "conditions": {}}
    for variant in args.variants:
        vals, tests, shuffleds = [], [], []
        for seed in seeds:
            model, stats, val_f1, _ = M.load_run(results / f"{variant}_s{seed}", args.device)
            X, y = load_split("test", stats)[:2]
            tests.append(float(f1_score(y, ET.predict(model, X, args.device), average="macro")))
            vals.append(val_f1)
            if variant == "simplex":
                sp = ET.predict_shuffled(model, X, args.device, shuffle_seed=ET.SHUFFLE_SEED + seed)
                shuffleds.append(float(f1_score(y, sp, average="macro")))
            del model
            torch.cuda.empty_cache()
        rec = {"val": vals, "test": tests,
               "val_mean": float(np.mean(vals)), "val_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
               "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0}
        out["conditions"][variant] = rec
        print(f"{variant:10s} val {rec['val_mean']:.4f}±{rec['val_std']:.4f} "
              f"test {rec['test_mean']:.4f}±{rec['test_std']:.4f}")
        if shuffleds:
            out["conditions"]["shuffled"] = {
                "test": shuffleds, "test_mean": float(np.mean(shuffleds)),
                "test_std": float(np.std(shuffleds, ddof=1)) if len(shuffleds) > 1 else 0.0}
            print(f"{'shuffled':10s} test {np.mean(shuffleds):.4f}±{out['conditions']['shuffled']['test_std']:.4f}")

    (results / "seed_summary.json").write_text(json.dumps(out, indent=1))
    print("wrote", results / "seed_summary.json")


if __name__ == "__main__":
    main()
