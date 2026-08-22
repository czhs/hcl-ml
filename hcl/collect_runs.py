#!/usr/bin/env python3
"""Index every finished run under <results>: best val, best epoch, parameter
count and the per-epoch curves. Writes <results>/all_runs.json (this is how
reference/all_runs.json — the source of the ablation numbers — was produced)."""
import argparse
import json
from pathlib import Path

from .paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    out = {}
    for d in sorted(p for p in results.iterdir() if (p / "summary.json").exists()):
        s = json.loads((d / "summary.json").read_text())
        recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        s["val_curve"] = [r["val_macro_f1"] for r in recs]
        s["train_loss_curve"] = [round(r["train_loss"], 4) for r in recs]
        if recs and "w_entropy_mean" in recs[0]:
            s["w_entropy_curve"] = [round(r["w_entropy_mean"], 3) for r in recs]
            s["w_argmax_agrees_curve"] = [round(r["w_argmax_agrees_label"], 3) for r in recs]
        out[d.name] = s
        print(f"{d.name:34s} seed {s['seed']} val {s['best_val_macro_f1']:.4f} ep {s['best_epoch']:2d} "
              f"params {s['trainable_params']:,}")
    (results / "all_runs.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {results / 'all_runs.json'} ({len(out)} runs)")


if __name__ == "__main__":
    main()
