#!/usr/bin/env bash
# Reproduce the three challenge conditions for one training seed (default 0).
# ~5 minutes on an RTX 4090 after setup.sh. Writes results/<run>/ and
# results/test_results.json, then prints the results table.
#
#   bash reproduce.sh            # seed 0
#   SEED=3 bash reproduce.sh     # another seed
#   SKIP_SUPPLEMENTARY=1 ...     # only direct + simplex (+ shuffled)
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
SEED=${SEED:-0}

echo "=== splits (subject-wise, split seed 42) ==="
python -m hcl.make_splits

echo "=== anchor tokens: verify configs/label_tokens.json regenerates ==="
python -m hcl.label_tokens --check || echo "WARNING: anchor profiles did not regenerate identically (see above); continuing with the committed configs/label_tokens.json"

echo "=== train: direct classifier (seed $SEED) ==="
python -m hcl.train --variant direct --seed "$SEED"
echo "=== train: context-embedding model, simplex projector (seed $SEED) ==="
python -m hcl.train --variant simplex --seed "$SEED"
if [ -z "${SKIP_SUPPLEMENTARY:-}" ]; then
  echo "=== train (supplementary): unconstrained projector ==="
  python -m hcl.train --variant free --seed "$SEED"
  echo "=== train (supplementary): simplex bottleneck without the LM ==="
  python -m hcl.train --variant bottleneck --seed "$SEED"
fi

echo "=== test evaluation: direct / simplex / shuffled (one pass each) ==="
python -m hcl.eval_test --seed "$SEED" --variants direct simplex free bottleneck
python -m hcl.plots --seed "$SEED"
python -m hcl.report
