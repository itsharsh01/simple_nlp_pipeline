# Simple NLP Pipeline

End-to-end MLOps project for **movie-review sentiment analysis** (positive / negative). It trains a lightweight scikit-learn model on IMDB data, tracks experiments with MLflow, and supports **local development** plus **Azure ML** training and deployment via Azure DevOps.

For a full walkthrough (diagrams, stage-by-stage detail, CI/CD, troubleshooting), see **[flow.md](./flow.md)**.

---

## What it does

| Step | Description |
|------|-------------|
| **Prepare** | Download [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) via Hugging Face → `data/train.csv`, `data/test.csv` |
| **Train** | TF-IDF + Logistic Regression, logged to MLflow |
| **Evaluate** | Test metrics + quality gate (default: accuracy & F1 ≥ 0.85) |
| **Register** | MLflow Model Registry + Azure ML model registry |
| **Serve** | Local FastAPI API, or Azure ML managed online endpoint |

**Model:** scikit-learn `Pipeline` (not deep learning) — fast to train on CPU.

---

## Quick start (local)

```powershell
cd simple_nlp_pipeline
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run stages 1–4 (prepare → train → evaluate → register)
python pipeline.py --no-serve

# Full run including API on http://localhost:8000
python pipeline.py
```

**API docs:** `http://localhost:8000/docs` after starting with `pipeline.py`.

Example prediction:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"This movie was absolutely fantastic!\"}"
```

---

## Project structure

```
simple_nlp_pipeline/
├── src/                    # Core logic (used locally and on Azure ML)
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── register.py
│   ├── train_and_register.py   # Azure ML job entrypoint
│   └── score.py                # Azure endpoint scoring
├── pipeline.py             # Local orchestrator
├── azure-pipeline.yml      # Azure DevOps CI/CD
├── training/               # Azure ML YAML (job, endpoint, deployment)
├── conda.yaml              # Environment for AML jobs & deployments
├── data/                   # Generated CSVs
├── serve/                  # Local FastAPI app
└── flow.md                 # Detailed architecture & pipeline docs
```

---

## Azure resources

| Resource | Default name |
|----------|----------------|
| Resource group | `mlops-wsh-rg-1` |
| Region | `centralus` |
| ML workspace | `mlops-workspace` |
| Compute cluster | `nlp-cluster` |
| Model | `sentiment_classifier` |
| Online endpoint | `nlp-sentiment-endpoint` |
| Deployment | `blue` |

Copy-paste CLI commands: **[commands.txt](./commands.txt)**.

### Submit a training job

```powershell
az ml job create `
  --file training/train_job.yml `
  --resource-group mlops-wsh-rg-1 `
  --workspace-name mlops-workspace `
  --stream
```

### One-time: create endpoint (before CI deploy)

```powershell
az ml online-endpoint create -f training/online_endpoint.yml -g mlops-wsh-rg-1 -w mlops-workspace
az ml online-deployment create -f training/online_deployment.yml -g mlops-wsh-rg-1 -w mlops-workspace --all-traffic
```

---

## Azure DevOps CI/CD

Pipeline file: **`azure-pipeline.yml`**

**Trigger:** pushes to `main` that change `src/`, `training/`, or `conda.yaml`.

| Stage | Action |
|-------|--------|
| **Train** | `az ml job create` → runs `train_and_register.py` on `nlp-cluster` |
| **Deploy** | Updates online deployment with latest `sentiment_classifier` version |

**Variable group** `mlops-vars` (Library):

- `RESOURCE_GROUP` = `mlops-wsh-rg-1`
- `AML_WORKSPACE` = `mlops-workspace`
- `MODEL_NAME` = `sentiment_classifier`
- `ENDPOINT_NAME` = `nlp-sentiment-endpoint`
- `DEPLOYMENT_NAME` = `blue`

Service connection name in YAML: `azure-service-connection`.

---

## Configuration

| Item | Default |
|------|---------|
| Training samples | 5,000 train / 1,000 test |
| MLflow experiment | `sentiment_analysis` |
| Quality gate | accuracy ≥ 0.85, F1 ≥ 0.85 |
| Registered model name | `sentiment_classifier` |

Override locally:

```powershell
python pipeline.py --train-size 10000 --min-accuracy 0.87 --no-serve
```

---

## Requirements

- Python 3.10+
- Internet on first run (Hugging Face dataset download)
- Azure CLI + `ml` extension (for cloud jobs / deployment)
- Azure DevOps (optional, for CI/CD)

Dependencies: **[requirements.txt](./requirements.txt)** (local) and **[conda.yaml](./conda.yaml)** (Azure ML).

---

## Documentation

| Doc | Contents |
|-----|----------|
| [flow.md](./flow.md) | Complete pipeline flow, Mermaid diagrams, MLflow vs AML registry, troubleshooting |
| [commands.txt](./commands.txt) | Azure CLI reference |

---

## License

Use and modify as needed for learning and MLOps demos.
