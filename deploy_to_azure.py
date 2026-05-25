"""
Train, register, and deploy the NLP sentiment model to Azure ML.

Mirrors the flow in nlp_train_register.ipynb, then deploys to a managed online endpoint.

Usage (from project root):
    python deploy_to_azure.py
    python deploy_to_azure.py --skip-train    # deploy latest registered model only

Requires: az login (or DefaultAzureCredential), pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# ── Config (edit these) ───────────────────────────────────────────────────────
SUBSCRIPTION_ID = "93708c7d-b32b-4d4c-ab77-bf578e0a7d4b"
RESOURCE_GROUP = "nlp-new-rg"
WORKSPACE_NAME = "nlp-ws"
MODEL_NAME = "nlp-sentiment-model"
EXPERIMENT_NAME = "nlp-sentiment-experiment"
ENDPOINT_NAME = "nlp-sentiment-endpoint"
DEPLOYMENT_NAME = "blue"
INSTANCE_TYPE = "Standard_DS2_v2"
INSTANCE_COUNT = 1


def _get_subscription_id() -> str:
    if SUBSCRIPTION_ID:
        return SUBSCRIPTION_ID
    out = subprocess.check_output(
        ["az", "account", "show", "--query", "id", "-o", "tsv"],
        text=True,
    )
    return out.strip()


def _get_credential():
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    try:
        cred = DefaultAzureCredential()
        cred.get_token("https://management.azure.com/.default")
        log.info("Using DefaultAzureCredential")
        return cred
    except Exception:
        log.info("DefaultAzureCredential failed — opening browser login")
        return InteractiveBrowserCredential()


def connect():
    from azure.ai.ml import MLClient

    ml_client = MLClient(
        credential=_get_credential(),
        subscription_id=_get_subscription_id(),
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    ws = ml_client.workspaces.get(WORKSPACE_NAME)
    log.info("Connected to workspace: %s", ws.name)
    log.info("MLflow URI: %s", ws.mlflow_tracking_uri)
    return ml_client, ws


def train_and_register(ws) -> float:
    """Notebook cells 4–6: sample data, train, mlflow log + register."""
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = {
        "text": [
            "I love this product, it works great!",
            "Amazing quality, highly recommend.",
            "Excellent service and fast delivery.",
            "Very happy with my purchase.",
            "This is the best thing I have bought.",
            "Outstanding performance and great value.",
            "Fantastic! Will definitely buy again.",
            "Superb quality, exceeded my expectations.",
            "Really impressed with this product.",
            "Great experience from start to finish.",
            "Terrible product, broke after one day.",
            "Very disappointed with the quality.",
            "Waste of money, do not buy.",
            "Worst purchase I have ever made.",
            "Stopped working after a week.",
            "Poor quality and slow delivery.",
            "Not worth the price at all.",
            "Awful experience, never again.",
            "Complete junk, highly disappointed.",
            "Defective product and bad customer service.",
            "Product is okay, nothing special.",
            "Average quality for the price.",
            "It works as described, nothing more.",
            "Decent product but could be better.",
            "Neither good nor bad, just average.",
        ],
        "label": ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 5,
    }

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )
    log.info("Train: %d | Test: %d", len(X_train), len(X_test))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log.info("Accuracy: %.4f", accuracy)
    log.info("\n%s", classification_report(y_test, y_pred))

    with mlflow.start_run(run_name="nlp-tfidf-logreg") as run:
        mlflow.log_param("model_type", "TF-IDF + LogisticRegression")
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        log.info("MLflow run id: %s", run.info.run_id)

    log.info("Model '%s' registered in workspace.", MODEL_NAME)
    return accuracy


def get_latest_model_version(ml_client) -> str:
    versions = list(ml_client.models.list(name=MODEL_NAME))
    if not versions:
        raise RuntimeError(f"No model named '{MODEL_NAME}' in workspace. Run training first.")
    latest = max(versions, key=lambda m: int(m.version))
    log.info("Latest model: %s version %s", latest.name, latest.version)
    return latest.version


def deploy(ml_client, model_version: str) -> None:
    """Create or update managed online endpoint + deployment."""
    from azure.ai.ml.entities import (
        CodeConfiguration,
        Environment,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
    )
    from azure.core.exceptions import ResourceNotFoundError

    model_ref = f"azureml:{MODEL_NAME}:{model_version}"
    conda_file = str(ROOT / "conda.yaml")
    code_path = str(ROOT / "src")

    if not Path(conda_file).exists():
        raise FileNotFoundError(f"Missing {conda_file}")
    if not Path(code_path, "score.py").exists():
        raise FileNotFoundError(f"Missing {code_path}/score.py")

    # Endpoint
    try:
        ml_client.online_endpoints.get(ENDPOINT_NAME)
        log.info("Endpoint '%s' already exists.", ENDPOINT_NAME)
    except ResourceNotFoundError:
        log.info("Creating endpoint '%s' …", ENDPOINT_NAME)
        endpoint = ManagedOnlineEndpoint(name=ENDPOINT_NAME, auth_mode="key")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        log.info("Endpoint created.")

    environment = Environment(
        conda_file=conda_file,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model_ref,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=code_path,
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
    )

    log.info(
        "Deploying %s to %s/%s (this can take 10–20 minutes) …",
        model_ref, ENDPOINT_NAME, DEPLOYMENT_NAME,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% traffic to this deployment
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    log.info("Deployment succeeded.")


def print_endpoint_info(ml_client) -> None:
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

    scoring_uri = endpoint.scoring_uri
    log.info("Scoring URI : %s", scoring_uri)
    log.info("Swagger     : %s/swagger.json", scoring_uri)
    if keys.primary_key:
        log.info("Primary key : %s", keys.primary_key[:8] + "…")

    print("\n--- Test with curl (PowerShell) ---")
    print(
        f'curl -X POST "{scoring_uri}" '
        f'-H "Content-Type: application/json" '
        f'-H "Authorization: Bearer <PRIMARY_KEY>" '
        f'-d \'{{"text": "I love this product!"}}\''
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, register, and deploy to Azure ML")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; deploy the latest registered model version only",
    )
    args = parser.parse_args()

    ml_client, ws = connect()

    if not args.skip_train:
        train_and_register(ws)
    else:
        import mlflow
        mlflow.set_tracking_uri(ws.mlflow_tracking_uri)

    version = get_latest_model_version(ml_client)
    deploy(ml_client, version)
    print_endpoint_info(ml_client)
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
