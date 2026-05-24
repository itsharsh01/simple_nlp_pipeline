"""Register model in MLflow and Azure ML workspace model registry."""

import argparse
import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from config import DEFAULT_MODEL_NAME, DEFAULT_EXPERIMENT, MODEL_OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _register_azure_ml_model(model_dir: Path, model_name: str) -> str | None:
    """Create/update model in AML registry (used by azure-pipeline.yml deploy stage)."""
    if not model_dir.is_dir() or not (model_dir / "MLmodel").exists():
        log.warning("No MLflow model dir at %s — skipping AML model create", model_dir)
        return None

    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Model
        from azure.identity import DefaultAzureCredential
    except ImportError:
        log.warning("azure-ai-ml not installed — skipping AML workspace model registration")
        return None

    sub = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")

    if sub and rg and ws:
        client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=sub,
            resource_group_name=rg,
            workspace_name=ws,
        )
    else:
        try:
            client = MLClient.from_config(credential=DefaultAzureCredential())
        except Exception as exc:
            log.warning("Could not create MLClient (local run?): %s", exc)
            return None

    aml_model = Model(
        path=str(model_dir),
        name=model_name,
        type=AssetTypes.MLFLOW_MODEL,
        description="TF-IDF + LogisticRegression sentiment classifier",
    )
    result = client.models.create_or_update(aml_model)
    log.info("Azure ML model registered: %s version %s", model_name, result.version)
    return str(result.version)


def register(
    model_name: str = DEFAULT_MODEL_NAME,
    experiment: str = DEFAULT_EXPERIMENT,
    model_dir: Path | None = None,
    run_id: str | None = None,
) -> str:
    """Register latest MLflow run and write deployable artifacts for Azure ML."""
    mlflow.set_experiment(experiment)
    client = MlflowClient()

    if not run_id:
        runs = mlflow.search_runs(
            experiment_names=[experiment],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError(f"No runs in experiment '{experiment}'.")
        run_id = runs.iloc[0]["run_id"]
        run_acc = runs.iloc[0].get("metrics.test_accuracy", "N/A")
        run_f1 = runs.iloc[0].get("metrics.test_f1", "N/A")
    else:
        run_acc = run_f1 = "N/A"

    model_uri = f"runs:/{run_id}/model"
    log.info("MLflow register_model %s from %s", model_name, model_uri)
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version

    for _ in range(30):
        mv = client.get_model_version(name=model_name, version=version)
        if mv.status == "READY":
            break
        time.sleep(2)

    # Export MLflow package for endpoint deployment & az ml model list
    out_dir = model_dir or MODEL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        mlflow.sklearn.load_model(model_uri),
        path=str(out_dir),
    )
    log.info("Model package written → %s", out_dir)

    aml_version = _register_azure_ml_model(out_dir, model_name)
    if aml_version:
        log.info("Deploy stage can use: azureml:%s:%s", model_name, aml_version)

    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register sentiment model")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    register(model_name=args.model_name, experiment=args.experiment)
