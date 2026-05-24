"""
Scoring script for Azure ML managed online endpoint.
Expects JSON: {"inputs": ["text1", "text2"]} or {"text": "single review"}
"""

import json
import logging
import os

import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "positive"}
_model = None


def init() -> None:
    global _model
    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model")
    if not os.path.isdir(model_dir):
        model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    log.info("Loading model from %s", model_dir)
    _model = mlflow.sklearn.load_model(model_dir)


def run(raw_data: str) -> dict:
    payload = json.loads(raw_data)
    texts = payload.get("inputs") or payload.get("texts")
    if texts is None and "text" in payload:
        texts = [payload["text"]]
    if isinstance(texts, str):
        texts = [texts]

    probas = _model.predict_proba(texts)
    predictions = []
    for text, proba in zip(texts, probas):
        cls = int(proba.argmax())
        predictions.append({
            "text": text,
            "label": LABEL_MAP[cls],
            "confidence": round(float(proba[cls]), 4),
        })

    return {"predictions": predictions}
