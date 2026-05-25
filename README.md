# Simple NLP Pipeline

Movie-review **sentiment analysis** (positive / negative / neutral) using scikit-learn, MLflow, and an **Azure ML managed online endpoint** for production inference.

For detailed architecture notes, see **[flow.md](./flow.md)**.

---

## What it does

| Step | Description |
|------|-------------|
| **Prepare** | Download [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) → CSVs (local pipeline) |
| **Train** | TF-IDF + Logistic Regression |
| **Evaluate** | Test metrics + quality gate (local pipeline) |
| **Register** | MLflow + Azure ML model registry |
| **Deploy** | `deploy_to_azure.py` → managed online endpoint |

**Model:** scikit-learn `Pipeline` — fast on CPU.

---

## Deploy to Azure (main workflow)

Train, register, and deploy in one command (same idea as `nlp_train_register.ipynb`):

**Workspace:** `nlp-ws` · **Resource group:** `nlp-new-rg`

```powershell
cd simple_nlp_pipeline
.\.venv\Scripts\activate
pip install -r requirements.txt
az login

python deploy_to_azure.py
```

Redeploy the latest registered model without retraining:

```powershell
python deploy_to_azure.py --skip-train
```

Config: edit the top of **`deploy_to_azure.py`**.

### Live endpoint

**Scoring URL:** https://nlp-ws-oiynf.eastus.inference.ml.azure.com/score

```powershell
curl -X POST "https://nlp-ws-oiynf.eastus.inference.ml.azure.com/score" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer <PRIMARY_KEY>" `
  -d '{"text": "I love this product!"}'
```

Get `<PRIMARY_KEY>` from Azure ML Studio → **Endpoints** → `nlp-sentiment-endpoint` → **Consume**.

---

## Local development (optional)

```powershell
python pipeline.py --no-serve    # prepare → train → evaluate → register
python pipeline.py               # same + FastAPI on http://localhost:8000
```

Local API docs: `http://localhost:8000/docs`

---

## Project structure

```
simple_nlp_pipeline/
├── deploy_to_azure.py      # Train, register, deploy to Azure ML
├── pipeline.py             # Local multi-stage pipeline
├── src/                    # Core scripts + score.py for endpoint
├── training/               # Optional AML YAML (endpoint / deployment)
├── conda.yaml              # Environment for Azure deployment
├── serve/                  # Local FastAPI
└── flow.md
```

---

## Azure resources

| Resource | Name |
|----------|------|
| Resource group | `nlp-new-rg` |
| ML workspace | `nlp-ws` |
| Model | `nlp-sentiment-model` |
| Online endpoint | `nlp-sentiment-endpoint` |
| Deployment | `blue` |

CLI helpers: **[commands.txt](./commands.txt)**

---

## Requirements

- Python 3.10+
- `az login` for Azure deployment
- **requirements.txt** (local) · **conda.yaml** (Azure endpoint environment)

---

## License

Use and modify as needed for learning and MLOps demos.
