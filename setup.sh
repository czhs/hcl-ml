#!/usr/bin/env bash
# One-shot, idempotent environment setup:
#   [1] virtualenv + dependencies   [2] UCI HAR dataset   [3] SmolLM2-360M-Instruct   [4] sanity check
#
# Environment variables (all optional):
#   PYTHON       interpreter used to create the venv        (default: python3)
#   VENV         venv location                               (default: .venv)
#   TORCH_SPEC   pip spec for torch, e.g. "torch==2.5.1"     (default: "torch>=2.5")
#   TORCH_INDEX  extra --index-url for the torch wheel, e.g. https://download.pytorch.org/whl/cu121
#   HCL_DATA     dataset root                                (default: ./data)
#   HCL_LM       HF id or local dir of the LM                (default: HuggingFaceTB/SmolLM2-360M-Instruct)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}
DATA=${HCL_DATA:-$PWD/data}
LM=${HCL_LM:-HuggingFaceTB/SmolLM2-360M-Instruct}
UCI_URL="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

echo "=== [1/4] virtualenv + dependencies ==="
[ -f "$VENV/bin/activate" ] || "$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install -q --upgrade pip
if [ -n "${TORCH_INDEX:-}" ]; then
  pip install -q "${TORCH_SPEC:-torch>=2.5}" --index-url "$TORCH_INDEX"
else
  pip install -q "${TORCH_SPEC:-torch>=2.5}"
fi
pip install -q -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== [2/4] UCI HAR dataset -> $DATA ==="
mkdir -p "$DATA"
if [ ! -d "$DATA/UCI HAR Dataset/train/Inertial Signals" ]; then
  ( cd "$DATA"
    [ -f uci_har.zip ] || curl -L -o uci_har.zip "$UCI_URL"
    unzip -o -q uci_har.zip
    [ -f "UCI HAR Dataset.zip" ] && unzip -o -q "UCI HAR Dataset.zip"   # the archive nests a second zip
    rm -f "UCI HAR Dataset.zip" )
fi
ls "$DATA/UCI HAR Dataset/train/Inertial Signals" | head -3

echo "=== [3/4] language model: $LM ==="
if [ -d "$LM" ]; then
  echo "using local directory"
else
  python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$LM", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("cached at", p)
PY
fi

echo "=== [4/4] sanity check ==="
HCL_DATA="$DATA" HCL_LM="$LM" python -m hcl.sanity
echo "SETUP DONE — activate with: source $VENV/bin/activate"
