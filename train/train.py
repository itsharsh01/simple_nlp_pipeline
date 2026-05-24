"""Thin wrapper — implementation lives in src/train.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import train  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--experiment", type=str, default="sentiment_analysis")
    args = parser.parse_args()
    train(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        max_iter=args.max_iter,
        C=args.C,
        experiment=args.experiment,
    )
