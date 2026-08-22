"""Single place for every filesystem location. Override with environment
variables so the same code runs on any machine:

  HCL_DATA     dataset root (default: <repo>/data)
  HCL_RESULTS  where runs are written (default: <repo>/results)
  HCL_LM       Hugging Face id or local directory of the frozen LM
               (default: HuggingFaceTB/SmolLM2-360M-Instruct — downloaded to
               the HF cache on first use)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATA = Path(os.environ.get("HCL_DATA", REPO / "data")).expanduser()
RAW = DATA / "UCI HAR Dataset"          # the official archive, unpacked
PROC = DATA / "processed"               # train/val/test .npz + splits.json

RESULTS = Path(os.environ.get("HCL_RESULTS", REPO / "results")).expanduser()

LM_ID = os.environ.get("HCL_LM", "HuggingFaceTB/SmolLM2-360M-Instruct")

LABEL_TOKENS = REPO / "configs" / "label_tokens.json"
REFERENCE = REPO / "reference"

CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]
