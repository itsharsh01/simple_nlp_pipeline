"""Train TF-IDF + Logistic Regression and log to MLflow."""

import argparse
import logging
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from config import DATA_DIR, MODEL_PKL, DEFAULT_EXPERIMENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train(
    max_features: int = 30_000,
    ngram_max: int = 2,
    max_iter: int = 500,
    C: float = 1.0,
    experiment: str = DEFAULT_EXPERIMENT,
) -> tuple[str, Pipeline]:
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run prepare_data first.")

    df = pd.read_csv(train_path)
    X_train = df["text"].astype(str).tolist()
    y_train = df["sentiment"].tolist()

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="tfidf_logreg") as run:
        run_id = run.info.run_id
        log.info("MLflow run id: %s", run_id)

        mlflow.log_params({
            "max_features": max_features,
            "ngram_range": f"(1,{ngram_max})",
            "C": C,
            "max_iter": max_iter,
            "solver": "lbfgs",
        })

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, ngram_max),
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\w{2,}",
            )),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", n_jobs=-1)),
        ])

        log.info("Training … (n=%d samples)", len(X_train))
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred)

        mlflow.log_metrics({
            "train_accuracy": round(train_acc, 4),
            "train_f1": round(train_f1, 4),
            "train_time_sec": round(elapsed, 2),
        })
        log.info("Train accuracy: %.4f | F1: %.4f", train_acc, train_f1)

        MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PKL, "wb") as f:
            pickle.dump(pipeline, f)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(str(MODEL_PKL), artifact_path="model_pickle")

    return run_id, pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    train(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        max_iter=args.max_iter,
        C=args.C,
        experiment=args.experiment,
    )
