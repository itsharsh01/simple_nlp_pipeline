"""Evaluate model on test set and enforce quality gate."""

import argparse
import json
import logging
import pickle
import sys

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import DATA_DIR, MODEL_PKL, EVAL_REPORT, DEFAULT_EXPERIMENT, DEFAULT_MIN_ACCURACY, DEFAULT_MIN_F1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate(
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
    min_f1: float = DEFAULT_MIN_F1,
    experiment: str = DEFAULT_EXPERIMENT,
) -> dict:
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found.")
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"{MODEL_PKL} not found.")

    df = pd.read_csv(test_path)
    X_test = df["text"].astype(str).tolist()
    y_test = df["sentiment"].tolist()

    with open(MODEL_PKL, "rb") as f:
        pipeline = pickle.load(f)

    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_recall": round(recall_score(y_test, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_pred_prob), 4),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report_str = classification_report(y_test, y_pred, target_names=["negative", "positive"])
    log.info("\n%s", report_str)

    mlflow.set_experiment(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])
    if not runs.empty:
        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")

    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {**metrics, "confusion_matrix": cm, "classification_report": report_str}
    with open(EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if metrics["test_accuracy"] < min_accuracy or metrics["test_f1"] < min_f1:
        log.error("Quality gate FAILED (accuracy=%.4f, f1=%.4f)", metrics["test_accuracy"], metrics["test_f1"])
        sys.exit(1)

    log.info("Quality gate PASSED")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    evaluate(min_accuracy=args.min_accuracy, min_f1=args.min_f1, experiment=args.experiment)
