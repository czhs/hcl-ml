#!/usr/bin/env bash
# Multi-seed protocol behind the mean ± std numbers: 8 training seeds x 4
# variants (split fixed at seed 42), then test-evaluate every val-selected
# checkpoint once. ~40 min on an RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
for s in $SEEDS; do
  for v in direct bottleneck simplex free; do
    echo "=== seed $s $v ==="
    python -m hcl.train --variant "$v" --seed "$s"
  done
done
python -m hcl.eval_seeds
python -m hcl.collect_runs
python -m hcl.report
echo "SEED SWEEP DONE"
