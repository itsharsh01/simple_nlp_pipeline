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

_model = None
_classes = None


def _get_classes(model):
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return model.named_steps["clf"].classes_
    return getattr(model, "classes_", None)


def init() -> None:
    global _model, _classes
    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model")
    if not os.path.isdir(model_dir):
        model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    log.info("Loading model from %s", model_dir)
    _model = mlflow.sklearn.load_model(model_dir)
    _classes = _get_classes(_model)
    log.info("Model classes: %s", _classes)


def run(raw_data: str) -> dict:
    payload = json.loads(raw_data)
    texts = payload.get("inputs") or payload.get("texts")
    if texts is None and "text" in payload:
        texts = [payload["text"]]
    if isinstance(texts, str):
        texts = [texts]

    pred_labels = _model.predict(texts)
    probas = _model.predict_proba(texts)
    classes = list(_classes) if _classes is not None else list(range(len(probas[0])))

    predictions = []
    for text, pred, proba in zip(texts, pred_labels, probas):
        idx = list(classes).index(pred) if pred in classes else int(proba.argmax())
        label = str(pred) if not isinstance(pred, (int, float)) else str(classes[idx])
        predictions.append({
            "text": text,
            "label": label,
            "confidence": round(float(proba[idx]), 4),
        })

    return {"predictions": predictions}
