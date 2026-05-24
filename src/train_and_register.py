"""
Azure ML training entrypoint (azure-pipeline.yml → training/train_job.yml).

Runs: prepare → train → evaluate → register (MLflow + AML model registry).
"""

import argparse
import logging
import os
import sys

# Ensure src/ is on path when run as `python train_and_register.py` from job cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_EXPERIMENT, DEFAULT_MODEL_NAME, DEFAULT_MIN_ACCURACY, DEFAULT_MIN_F1
from evaluate import evaluate
from prepare_data import prepare
from register import register
from train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register NLP model on Azure ML")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    args = parser.parse_args()

    log.info("=== 1/4 Prepare data ===")
    prepare(train_size=args.train_size, test_size=args.test_size, seed=args.seed)

    log.info("=== 2/4 Train ===")
    run_id, _ = train(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        max_iter=args.max_iter,
        C=args.C,
        experiment=args.experiment,
    )

    log.info("=== 3/4 Evaluate ===")
    evaluate(
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
        experiment=args.experiment,
    )

    log.info("=== 4/4 Register ===")
    register(model_name=args.model_name, experiment=args.experiment, run_id=run_id)
    log.info("Train and register completed successfully.")


if __name__ == "__main__":
    main()
