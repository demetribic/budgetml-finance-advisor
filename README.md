# BudgetML — ML-Powered Personal Finance Advisor

BudgetML analyzes a user's transaction history and returns actionable financial
guidance. It combines three PyTorch transformer models, a set of lightweight
"intelligence" detectors, and a deterministic rules engine, all served through a
FastAPI web app with a small built-in dashboard.

## What it does

- **Spending forecast** — transformer that projects near-term category spend.
- **Anomaly detection** — transformer (plus a contrastive variant) that flags
  unusual transactions.
- **Bulk-buy recommendations** — transformer that spots stock-up opportunities.
- **Intelligence layer** — cash-crunch prediction, goal inference, life-event and
  behavioral-bias detection, peer comparison.
- **Rules engine** — subscription analysis, time-value calculations, deal finding,
  and a decision engine that turns model output into ranked suggestions.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/<you>/budgetml-finance-advisor.git
cd budgetml-finance-advisor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Get the trained models (not stored in git — see Releases)
#    Download budgetml-models.tar from the latest release, then:
tar -xf budgetml-models.tar          # populates models/saved/

# 3. Get a database — pick ONE:
#    a) Seed a realistic 6-month demo user (recommended):
python scripts/seed_test_user.py
#    b) ...or download the prebuilt budgetml.db from the release into data/

# 4. Run the API + dashboard
uvicorn api.app:app --port 8000
```

Then open <http://localhost:8000>.

> The app auto-creates its SQLite schema on first run (`CREATE TABLE IF NOT
> EXISTS`), so step 3 only matters for having data to analyze.

## Releases & model weights

Trained weights are **not** committed to git (they're large binaries). Download
them from the [Releases page](../../releases):

| Asset | Contents |
|-------|----------|
| `budgetml-models.tar` | All trained models → extract into `models/saved/` |
| `budgetml.db`         | Prebuilt demo database (optional) → place in `data/` |

The release excludes the optional `local_llm/` base model used by
`models/intelligence/local_llm.py`; download that separately if you want the
LLM-backed narrative features.

## Project layout

```
api/         FastAPI app, persistence, explainer, static dashboard
config/      Settings loader + settings.yaml
data/        Loader, preprocessing, synthetic data, feature pipeline
models/      Transformers, embeddings, intelligence, pretraining, baselines, VAE
rules/       Decision engine + subscription / time-value / deal logic
training/    Training entrypoints for every model (+ SLURM scripts)
tests/       Unit tests
scripts/     seed_test_user.py — populate a demo user
```

## Training (optional)

Models are trained on an HPC/SLURM cluster:

```bash
sbatch train_all.slurm        # train everything
python -m training.train_forecast   # or train a single model locally
```

## Requirements

Python 3.10+ and the packages in `requirements.txt` (PyTorch, FastAPI, scikit-learn,
LightGBM/XGBoost, SetFit, etc.).
