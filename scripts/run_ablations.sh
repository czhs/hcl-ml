#!/usr/bin/env bash
# Every ablation reported in the technical note / writeup (val-only; seeds as
# noted). Run after reproduce.sh. ~1.5 h on an RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
train() { echo "=== $* ==="; python -m hcl.train "$@"; }

# Geometric (meaning-free, maximally separated) anchor ladder -> configs/label_tokens.json
python -m hcl.geometric_anchors --check || python -m hcl.geometric_anchors

for s in 0 1 2; do
  train --variant simplex --anchors geometric --seed $s                    # anchor semantics vs separation
  train --variant simplex --learn-anchors --seed $s                        # trainable corners (token init)
  train --variant simplex --learn-anchors --anchor-init random --seed $s   # trainable corners, off-manifold init
  train --variant simplex --lm-init random --seed $s                       # H0: random frozen transformer
  train --variant simplex --no-prompt --seed $s                            # H1: no prompt text at all
  train --variant simplex --prompt-style unrelated --seed $s               # H1.5: recipe prompt
done

# K-sweep: number of corners (frozen max-separated tokens), plus gated K=1
for K in 1 2 4 8 12 16 32 64; do
  train --variant simplex --anchors "geometric$K" --seed 0
done
train --variant simplex --anchors geometric1 --gate --seed 0

# Analyses on trained checkpoints
python -m hcl.mech_probe --run simplex_s1 || python -m hcl.mech_probe --run simplex_s0
python -m hcl.analyze_continuity --run simplex_s0
python -m hcl.collect_runs
python -m hcl.plots
echo "ABLATIONS DONE"
