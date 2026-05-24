"""Thin wrapper — implementation lives in src/evaluate.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import evaluate  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-f1", type=float, default=0.85)
    parser.add_argument("--experiment", type=str, default="sentiment_analysis")
    args = parser.parse_args()
    evaluate(min_accuracy=args.min_accuracy, min_f1=args.min_f1, experiment=args.experiment)
