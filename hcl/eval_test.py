#!/usr/bin/env python3
"""Final test-set evaluation — the three challenge conditions, one pass each.

  1. direct    — direct sensor classifier
  2. simplex   — context-embedding model (simplex over the balanced anchors)
  3. shuffled  — the SAME trained simplex model, its projected context
                 embeddings permuted between test examples (complete
                 embeddings, not individual values; no retraining)
Supplementary: free (unconstrained 960-d projector), bottleneck (no LM).

Reads <results>/<variant>_s<seed>/ and writes <results>/test_results.json
(macro-F1, per-class F1, confusion matrices).

  python -m hcl.eval_test --seed 0
  python -m hcl.eval_test --results reference/checkpoints --out results/reference_eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from . import model as M
from .data import load_split
from .paths import CLASSES, RESULTS

SHUFFLE_SEED = 123


@torch.no_grad()
def predict(model, X, device, bs=512):
    preds = []
    for i in range(0, len(X), bs):
        logits, _ = model(X[i:i + bs].to(device))
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds)


@torch.no_grad()
def predict_shuffled(model, X, device, shuffle_seed=SHUFFLE_SEED, bs=512):
    """Sensor-dependence check: compute every test example's context embedding,
    permute the embeddings across examples, run the frozen LM + head again."""
    es = []
    for i in range(0, len(X), bs):
        e, _ = model.context_embedding(model.encoder(X[i:i + bs].to(device)))
        es.append(e)
    E = torch.cat(es)                                                    # (N, d), test order
    perm = torch.from_numpy(np.random.default_rng(shuffle_seed).permutation(len(E)))
    E = E[perm]                                                          # whole embeddings swapped
    preds = []
    for i in range(0, len(E), bs):
        logits, _ = model(X[i:i + bs].to(device), context_override=E[i:i + bs])
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds)


def report(name, y, preds):
    y, preds = np.asarray(y), np.asarray(preds)
    f1 = f1_score(y, preds, average="macro")
    per = f1_score(y, preds, average=None, labels=list(range(6)))
    cm = confusion_matrix(y, preds, labels=list(range(6)))
    print(f"{name:22s} macro-F1 {f1:.4f}")
    return {"macro_f1": round(float(f1), 4),
            "per_class_f1": {c: round(float(v), 4) for c, v in zip(CLASSES, per)},
            "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS), help="directory holding <variant>_s<seed>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    ap.add_argument("--variants", nargs="+", default=["direct", "simplex", "free"])
    ap.add_argument("--out", default=None, help="default <results>/test_results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    out_path = Path(args.out).expanduser() if args.out else results / "test_results.json"

    out = {"seed": args.seed, "shuffle_seed": args.shuffle_seed, "classes": CLASSES, "conditions": {}}
    for variant in args.variants:
        run = results / f"{variant}_s{args.seed}"
        if not (run / "best.pt").exists():
            print(f"skip {variant}: no checkpoint at {run}")
            continue
        model, stats, val_f1, _ = M.load_run(run, args.device)
        X, y = load_split("test", stats)[:2]
        rec = report(variant, y, predict(model, X, args.device))
        rec.update(val_macro_f1=round(val_f1, 4), trainable_params=M.trainable_params(model), run=str(run))
        out["conditions"][variant] = rec
        if variant == "simplex":
            out["conditions"]["shuffled"] = report("simplex (shuffled)", y,
                                                   predict_shuffled(model, X, args.device, args.shuffle_seed))
        del model
        torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
