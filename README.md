# IPL Intelligence Engine

A full ML pipeline for IPL cricket analytics — ball-by-ball pressure modeling, toss alpha decay, and match result prediction.

## Project Structure

```
ipl_intelligence_engine/
├── data/
│   ├── loader.py               # Data loading & preprocessing utilities
│   └── kaggle_ipl.py           # Download & load IPL data from Kaggle
├── models/
│   ├── pressure_index.py       # Ball-by-ball pressure index + collapse predictor
│   ├── toss_alpha_decay.py     # Toss advantage decay by venue & era
│   ├── pre_match_features.py   # ELO, H2H, form, pre-match feature engineering
│   └── unified_predictor.py   # Stacked ensemble match result predictor
├── evaluation/
│   └── evaluate.py             # AUC, Brier, calibration, segment breakdown
├── backend/
│   ├── main.py                 # FastAPI — serves public/ + POST /api/predict
│   ├── inference.py            # Pickle stack (local) or lite exports (Vercel)
│   └── inference_lite.py       # numpy + lightgbm inference for serverless
├── api/
│   └── index.py                # Vercel serverless entry (Mangum → FastAPI)
├── public/
│   ├── index.html              # cricgnaan UI (calls /api/predict)
│   └── static/                 # styles.css, app.js
├── dashboard/
│   └── app.py                  # Streamlit live match dashboard (alternative UI)
├── notebooks/
│   └── full_pipeline.ipynb     # End-to-end walkthrough notebook
├── requirements.txt          # Vercel: numpy + lightgbm + FastAPI only (under ~250 MB install)
├── requirements-ui.txt       # Local: training + Streamlit + full ML stack
├── requirements-dev.txt      # Optional: Jupyter, Kaggle CLI, SHAP, …
└── README.md
```

## Quickstart

```bash
# Full local stack (Streamlit, FastAPI dev server, training deps):
pip install -r requirements-ui.txt
# Notebooks / Kaggle downloads / SHAP / matplotlib (optional):
# pip install -r requirements-dev.txt
```

### Option A: Use data from Kaggle (recommended)

1. Get a Kaggle API key: [Kaggle](https://www.kaggle.com) → Account → Create New API Token (downloads `kaggle.json`).
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).
3. Download and load the [IPL dataset (2008–2025)](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025/data):

```bash
# From project root
python -c "
from data.kaggle_ipl import load_ipl_from_kaggle
df = load_ipl_from_kaggle(download_if_missing=True)
print(df.head())
"
```

This downloads the dataset to `data/kaggle_ipl/` and returns a DataFrame ready for the pipeline. Column names from the Kaggle CSV are mapped to the expected schema (e.g. `batsman` → `batter`, `total_runs` → `runs_total`). If the dataset uses different column names, edit `KAGGLE_COLUMN_MAP` in `data/kaggle_ipl.py`.

### Option B: Use your own CSV

```bash
# 1. Load & preprocess your IPL ball-by-ball CSV
python -c "from data.loader import load_ipl; df = load_ipl('your_data.csv')"
```

### Train and evaluate

```bash
# 2. Build all features and train
python models/unified_predictor.py

# 3. Evaluate
python evaluation/evaluate.py

# 4. Launch dashboard
python3 -m streamlit run dashboard/app.py
```

### Web UI (cricgnaan — HTML + real models)

Uses the same ensemble as `dashboard/app.py`, served by FastAPI:

```bash
pip install -r requirements-ui.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/** — the page in **`public/`** calls **`POST /api/predict`**.

### Deploy on Vercel (HTML UI + API)

The repo includes **`vercel.json`** and **`api/index.py`** (Mangum wraps the FastAPI app). Vercel serves **`public/`** at the domain root; **`/api/*`** routes to the Python function.

1. Push the repo to GitHub and [import the project in Vercel](https://vercel.com/new).
2. **Lambda has a ~500 MB cap on installed dependencies.** Root **`requirements.txt`** installs only **FastAPI, Mangum, numpy, and LightGBM** (no pandas/sklearn/scipy). Inference uses **exported lite assets** next to your pickles — native **`lgb_*_model.txt`**, **`inference_calibrators.json`**, **`toss_alpha_table.json`**, **`team_stats.json`**. Generate them by training with **`python run_from_kaggle.py`** (writes these automatically) or, if you already have **`models/saved/*.pkl`**, run **`python -m models.inference_export models/saved`**. Commit those files to the repo Vercel builds from.
3. **Pickles alone are not enough on Vercel** — without the lite export files, the API would need pandas/sklearn and exceed the size limit. For local development, use **`pip install -r requirements-ui.txt`**.
4. If anything still fails cold-start or size checks, host the API on **Railway/Render** and point the static UI at that base URL.

**Streamlit** is unchanged: use **Streamlit Community Cloud** with `dashboard/app.py` (see below).

## Deploy the Streamlit dashboard

The Streamlit app is separate from the Vercel HTML UI. The easiest host is **[Streamlit Community Cloud](https://streamlit.io/cloud)** (free, connects to GitHub).

1. Push this repo to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io), **New app** → pick the repo, branch, and main file **`dashboard/app.py`**.
3. **Models:** If `models/saved/*.pkl` are not in the repo, pick one approach:
   - **Secrets (recommended for public repos):** After training locally, zip the files: `cd models/saved && zip -r ../../models_bundle.zip *.pkl`. Upload the zip to a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) (or any public HTTPS URL). In the Streamlit app **Settings → Secrets**, add `MODEL_BUNDLE_URL = "https://..."` pointing to that zip (see `.streamlit/secrets.toml.example`).
   - **Commit binaries:** If each file is under GitHub’s size limits, you can `git add -f models/saved/*.pkl` and push so the models ship with the repo.

In **Streamlit Community Cloud** → app **Settings**, set the **Python requirements file** to **`requirements-ui.txt`** (Streamlit, Plotly, and the full training stack). If your workspace only allows the default file, use a branch where `requirements.txt` matches `requirements-ui.txt`, or merge those dependencies manually.

The repo includes `runtime.txt` (Python 3.11), `.python-version` (3.11 for Vercel/local tooling), `packages.txt` (`libgomp1` for LightGBM), and `.streamlit/config.toml` for Cloud.

## Expected Accuracy

| Stage          | Accuracy | AUC  |
|----------------|----------|------|
| Pre-match      | ~67%     | 0.74 |
| Over 10        | ~76%     | 0.80 |
| Over 15        | ~84%     | 0.83 |
| Collapse alert | ~81%     | 0.85 |

## Data Requirements

Your IPL dataset should include these key columns:
`match_id, date, innings, batting_team, bowling_team, over, ball_no, batter, runs_batter, runs_total, wicket_kind, runs_target, toss_winner, toss_decision, match_won_by, venue, season, player_out, valid_ball`

Full 64-column schema supported as described in the conversation.

## Models

- **Pressure Index**: LightGBM classifier — P(collapse in next 30 balls)
- **Toss Alpha Decay**: Bayesian Beta posterior — toss edge per venue × era
- **Match Predictor**: Stacked ensemble (LightGBM base + Logistic meta-learner)
