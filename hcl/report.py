#!/usr/bin/env python3
"""Print the challenge results table (Markdown) from test_results.json and, if
present, seed_summary.json. Also writes <results>/RESULTS.md.

  python -m hcl.report                              # your run
  python -m hcl.report --results reference          # the numbers in the technical note
"""
import argparse
import json
from pathlib import Path

from .paths import RESULTS

ROWS = [("direct", "Direct sensor classifier"),
        ("simplex", "Context-embedding model (simplex)"),
        ("shuffled", "Context model, shuffled embeddings"),
        ("free", "*supplementary:* unconstrained 960-d projector"),
        ("bottleneck", "*supplementary:* simplex code without the LM")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    lines = []

    tr = results / "test_results.json"
    if tr.exists():
        d = json.loads(tr.read_text())
        conds = d["conditions"]
        lines += [f"### Single run (training seed {d.get('seed', 0)}, shuffle seed {d['shuffle_seed']})", "",
                  "| Condition | Macro-F1 (test) | Macro-F1 (val) | Trainable params |", "|---|---:|---:|---:|"]
        for key, name in ROWS:
            if key in conds:
                c = conds[key]
                lines.append(f"| {name} | **{c['macro_f1']:.4f}** | {c.get('val_macro_f1', '—')} | "
                             f"{c['trainable_params']:,} |" if "trainable_params" in c
                             else f"| {name} | **{c['macro_f1']:.4f}** | — | — |")
        lines.append("")

    ss = results / "seed_summary.json"
    if ss.exists():
        d = json.loads(ss.read_text())
        n = len(d["seeds"])
        lines += [f"### Mean ± std over training seeds {d['seeds']}", "",
                  "| Condition | Macro-F1 (test) | Macro-F1 (val) | per-seed test |", "|---|---:|---:|---|"]
        for key, name in ROWS:
            if key in d["conditions"]:
                c = d["conditions"][key]
                val = f"{c['val_mean']:.4f} ± {c['val_std']:.4f}" if "val_mean" in c else "—"
                per = " ".join(f"{t:.3f}" for t in c["test"])
                lines.append(f"| {name} | **{c['test_mean']:.4f} ± {c['test_std']:.4f}** | {val} | {per} |")
        s = d["conditions"].get("simplex")
        if s:   # a converged-seeds view: flag collapsed seeds (val < 0.8) explicitly
            ok = [(v, t) for v, t in zip(s["val"], s["test"]) if v >= 0.8]
            if 0 < len(ok) < n:
                import numpy as np
                t = np.array([x[1] for x in ok])
                lines.append(f"| — simplex, {len(ok)}/{n} converged seeds (val ≥ 0.8) | "
                             f"**{t.mean():.4f} ± {t.std(ddof=1):.4f}** | | |")
        lines.append("")

    ar = results / "all_runs.json"
    if ar.exists():
        import collections
        import numpy as np
        runs = json.loads(ar.read_text())
        groups = collections.defaultdict(list)
        for name, r in runs.items():
            if "_s" in name:
                groups[name.rsplit("_s", 1)[0]].append(r["best_val_macro_f1"])
        lines += ["### All runs (best validation macro-F1; ablations are val-only)", "",
                  "| run tag | seeds | val macro-F1 | per-seed |", "|---|---:|---:|---|"]
        for tag, vs in sorted(groups.items()):
            sd = f" ± {np.std(vs, ddof=1):.4f}" if len(vs) > 1 else ""
            lines.append(f"| `{tag}` | {len(vs)} | {np.mean(vs):.4f}{sd} | {' '.join(f'{v:.4f}' for v in vs)} |")
        lines.append("")

    if not lines:
        raise SystemExit(f"nothing to report in {results}")
    text = "\n".join(lines)
    print(text)
    (results / "RESULTS.md").write_text(text + "\n")
    print(f"(wrote {results / 'RESULTS.md'})")


if __name__ == "__main__":
    main()
