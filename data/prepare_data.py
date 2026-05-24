"""Thin wrapper — implementation lives in src/prepare_data.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prepare_data import prepare  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare IMDB data")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
