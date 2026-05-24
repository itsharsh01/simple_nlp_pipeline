"""
Local pipeline — same stages as src/, for development without Azure ML.
Azure DevOps uses azure-pipeline.yml → training/train_job.yml instead.
"""

import argparse
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")


def run_stage(name: str, script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, script] + (extra_args or [])
    log.info("=" * 60)
    log.info("STAGE: %s", name)
    log.info("CMD  : %s", " ".join(cmd))
    log.info("=" * 60)

    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        log.error("Stage '%s' failed — aborting.", name)
        sys.exit(result.returncode)
    log.info("Stage '%s' completed.\n", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local sentiment-analysis pipeline")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-f1", type=float, default=0.85)
    parser.add_argument("--model-name", type=str, default="sentiment_classifier")
    parser.add_argument("--experiment", type=str, default="sentiment_analysis")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-serve", action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    run_stage("1 — Prepare Data", os.path.join(SRC, "prepare_data.py"), [
        "--train-size", str(args.train_size),
        "--test-size", str(args.test_size),
        "--seed", str(args.seed),
    ])
    run_stage("2 — Train", os.path.join(SRC, "train.py"), [
        "--max-features", str(args.max_features),
        "--ngram-max", str(args.ngram_max),
        "--max-iter", str(args.max_iter),
        "--C", str(args.C),
        "--experiment", args.experiment,
    ])
    run_stage("3 — Evaluate", os.path.join(SRC, "evaluate.py"), [
        "--min-accuracy", str(args.min_accuracy),
        "--min-f1", str(args.min_f1),
        "--experiment", args.experiment,
    ])
    run_stage("4 — Register", os.path.join(SRC, "register.py"), [
        "--model-name", args.model_name,
        "--experiment", args.experiment,
    ])

    log.info("Stages 1–4 done in %.1fs", time.time() - t_start)

    if not args.no_serve:
        log.info("Starting FastAPI on http://localhost:%d", args.port)
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "serve.serve:app", "--host", "0.0.0.0", "--port", str(args.port)],
            cwd=HERE,
            check=True,
        )


if __name__ == "__main__":
    main()
