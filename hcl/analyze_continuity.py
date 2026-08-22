#!/usr/bin/env python3
"""How is the simplex code w actually used? (CPU-only; val split.)

  1. Softness: entropy percentiles + participation ratio (effective #corners).
  2. Dimensionality: PCA of w — how many of the 5 degrees of freedom carry variance.
  3. Discreteness: unique codes after rounding w to a 0.05 grid; within-class
     spread vs between-class centroid distance.
  4. Smoothness: consecutive same-subject windows overlap 50% in time — if the
     map is continuous their w should be much closer than random same-class pairs.
  5. Subject probe: within each class, logistic regression from the 6-d w alone
     to WHICH of the (unseen) val subjects produced the window, vs a
     shuffled-label control — does the code carry style/identity?

  python -m hcl.analyze_continuity --run simplex_s0
Writes <results>/continuity.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from . import model as M
from .data import load_norm_stats, load_split_raw
from .paths import CLASSES, RESULTS


def codes_for_run(run_dir):
    """Encoder + projector of a simplex run, on CPU, without loading the LM."""
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    net = M.BottleneckClassifier()
    net.load_state_dict({k: v for k, v in ck["state"].items() if k.startswith(("encoder.", "projector."))}, strict=False)
    net.eval()
    stats = load_norm_stats(run_dir)
    Xr, y, subj = load_split_raw("val")
    X = torch.from_numpy((Xr - stats[0]) / stats[1])
    with torch.no_grad():
        w = F.softmax(net.projector(net.encoder(X)), dim=-1).numpy()
    return w, y, subj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--run", default="simplex_s0")
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    torch.set_num_threads(4)
    w, y, subj = codes_for_run(results / args.run)

    H = -(w * np.log(w + 1e-9)).sum(1)
    pr = 1.0 / (w ** 2).sum(1)
    print(f"[1] entropy p10/50/90: {np.percentile(H,10):.2f}/{np.percentile(H,50):.2f}/{np.percentile(H,90):.2f} "
          f"nats (max {np.log(6):.2f}); participation ratio p10/50/90: "
          f"{np.percentile(pr,10):.2f}/{np.percentile(pr,50):.2f}/{np.percentile(pr,90):.2f} of 6")

    wc = w - w.mean(0)
    ev = np.linalg.svd(wc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    cum = np.cumsum(ev)
    print(f"[2] PCA var by dim: {np.round(ev[:5], 3)}; dims for 90/95/99%: "
          f"{int(np.searchsorted(cum, .90) + 1)}/{int(np.searchsorted(cum, .95) + 1)}/{int(np.searchsorted(cum, .99) + 1)} of 5")

    uniq = len(np.unique(np.round(w / 0.05).astype(int), axis=0))
    cents = np.stack([w[y == c].mean(0) for c in range(6)])
    within = np.mean([np.linalg.norm(w[y == c] - cents[c], axis=1).mean() for c in range(6)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j]) for i in range(6) for j in range(i + 1, 6)])
    print(f"[3] unique codes (0.05 grid): {uniq}/{len(w)} ({100 * uniq / len(w):.0f}%); within-class spread "
          f"{within:.3f} vs between-centroid dist {between:.3f} (ratio {within / between:.2f})")

    adj, rand = [], []
    rng = np.random.default_rng(0)
    for i in range(len(w) - 1):
        if subj[i] == subj[i + 1] and y[i] == y[i + 1]:
            adj.append(np.linalg.norm(w[i] - w[i + 1], ord=1))
            pool = np.where((y == y[i]) & (subj != subj[i]))[0]
            if len(pool):
                rand.append(np.linalg.norm(w[i] - w[rng.choice(pool)], ord=1))
    print(f"[4] |dw| L1, adjacent overlapping windows: {np.mean(adj):.3f} vs random same-class "
          f"cross-subject pairs: {np.mean(rand):.3f} (ratio {np.mean(adj) / np.mean(rand):.2f})")

    argmax_agree = float((w.argmax(1) == y).mean())
    probe, control = {}, {}
    for c in range(6):
        m = y == c
        if len(np.unique(subj[m])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000)
        probe[CLASSES[c]] = float(cross_val_score(clf, w[m], subj[m], cv=3).mean())
        control[CLASSES[c]] = float(cross_val_score(clf, w[m], rng.permutation(subj[m]), cv=3).mean())
    chance = 1.0 / len(np.unique(subj))
    print(f"[5] subject probe from w (chance {chance:.2f}): " + ", ".join(
        f"{k[:6]} {v:.2f} (ctrl {control[k]:.2f})" for k, v in probe.items()))
    print(f"    argmax(w) == label: {argmax_agree:.3f}")

    out = {"run": args.run, "entropy_p50": float(np.percentile(H, 50)), "pr_p50": float(np.percentile(pr, 50)),
           "pca_var": [float(v) for v in ev[:5]], "unique_codes_frac": uniq / len(w),
           "within_between_ratio": float(within / between), "adj_over_random": float(np.mean(adj) / np.mean(rand)),
           "argmax_agrees_label": argmax_agree, "subject_probe_acc": probe,
           "subject_probe_shuffled_control": control, "subject_chance": chance}
    (results / "continuity.json").write_text(json.dumps(out, indent=1))
    print("wrote", results / "continuity.json")


if __name__ == "__main__":
    main()
