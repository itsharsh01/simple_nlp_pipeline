"""Shared paths and defaults for local runs and Azure ML jobs."""

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
# Full repo locally; on AML only src/ is uploaded as code
ROOT = SRC_DIR.parent if (SRC_DIR.parent / "conda.yaml").exists() else SRC_DIR

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODEL_PKL = Path(os.environ.get(
    "MODEL_PKL",
    ROOT / "train" / "model.pkl" if (ROOT / "train").exists() else SRC_DIR / "model.pkl",
))
EVAL_REPORT = Path(os.environ.get(
    "EVAL_REPORT",
    ROOT / "evaluate" / "eval_report.json"
    if (ROOT / "evaluate").exists()
    else SRC_DIR / "eval_report.json",
))

MODEL_OUTPUT_DIR = Path(
    os.environ.get("AZUREML_OUTPUT_model_output", ROOT / "outputs" / "model")
)

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "sentiment_classifier")
DEFAULT_EXPERIMENT = os.environ.get("EXPERIMENT_NAME", "sentiment_analysis")
DEFAULT_MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.85"))
DEFAULT_MIN_F1 = float(os.environ.get("MIN_F1", "0.85"))
