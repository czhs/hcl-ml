#!/usr/bin/env python3
"""UMAP of the val windows' simplex codes (optional: pip install -r requirements-analysis.txt).

Each trajectory = a maximal run of consecutive windows with the same subject
and label (50%-overlapping slices of one continuous recording). The six pure
corners (one-hot w) are embedded in the same map. Writes
<results>/umap_<run>.json and <results>/figures/umap_<run>.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import model as M
from .analyze_continuity import codes_for_run
from .paths import CLASSES, RESULTS

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--run", default="simplex_s0")
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    import umap   # noqa: E402  (optional dependency)

    w, y, subj = codes_for_run(results / args.run)
    runs = np.zeros(len(w), dtype=int)
    for i in range(1, len(w)):
        runs[i] = runs[i - 1] + int(subj[i] != subj[i - 1] or y[i] != y[i - 1])
    print(f"{len(w)} windows, {runs[-1] + 1} runs")

    ck_args = __import__("torch").load(results / args.run / "best.pt", map_location="cpu", weights_only=False)["args"]
    names = [e_["token"].strip() for e_ in M.anchor_entries(ck_args.get("anchors_profile", ck_args.get("anchors", "balanced")))]
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(np.vstack([w, np.eye(6, dtype=np.float32)]))
    pts, cor = emb[:-6], emb[-6:]
    out = {"run": args.run,
           "points": [{"x": round(float(px), 3), "y": round(float(py), 3), "c": int(c), "run": int(r), "subj": int(sj)}
                      for (px, py), c, r, sj in zip(pts, y, runs, subj)],
           "corners": [{"x": round(float(px), 3), "y": round(float(py), 3), "name": n} for (px, py), n in zip(cor, names)]}
    (results / f"umap_{args.run}.json").write_text(json.dumps(out))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(6):
        m = y == c
        ax.scatter(pts[m, 0], pts[m, 1], s=4, alpha=0.6, label=CLASSES[c])
    for (px, py), n in zip(cor, names):
        ax.scatter(px, py, marker="*", s=160, color="k", zorder=5)
        ax.annotate(n, (px, py), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"UMAP of simplex codes w (val), {args.run}; stars = pure corners", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    (results / "figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(results / "figures" / f"umap_{args.run}.png", dpi=150)
    print("wrote", results / "figures" / f"umap_{args.run}.png")


if __name__ == "__main__":
    main()
