"""Thin wrapper — implementation lives in src/register.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from register import register  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register sentiment model")
    parser.add_argument("--model-name", type=str, default="sentiment_classifier")
    parser.add_argument("--experiment", type=str, default="sentiment_analysis")
    args = parser.parse_args()
    register(model_name=args.model_name, experiment=args.experiment)
