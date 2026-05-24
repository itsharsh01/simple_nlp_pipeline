"""Download IMDB and write train.csv / test.csv."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def prepare(train_size: int = 5_000, test_size: int = 1_000, seed: int = 42) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading IMDB dataset from HuggingFace …")
    dataset = load_dataset("stanfordnlp/imdb")

    train_df = (
        dataset["train"]
        .to_pandas()
        .sample(n=min(train_size, len(dataset["train"])), random_state=seed)
        .reset_index(drop=True)
    )
    test_df = (
        dataset["test"]
        .to_pandas()
        .sample(n=min(test_size, len(dataset["test"])), random_state=seed)
        .reset_index(drop=True)
    )

    train_df = train_df.rename(columns={"label": "sentiment"})
    test_df = test_df.rename(columns={"label": "sentiment"})
    train_df["sentiment_label"] = train_df["sentiment"].map({0: "negative", 1: "positive"})
    test_df["sentiment_label"] = test_df["sentiment"].map({0: "negative", 1: "positive"})

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info("Saved %d training rows → %s", len(train_df), train_path)
    log.info("Saved %d test rows → %s", len(test_df), test_path)
    return train_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IMDB data")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
