# Deploy to Git (e.g. GitHub)

The repo is set up to be git-ready: `.gitignore` excludes `.venv`, trained `.pkl` files, downloaded Kaggle data, and secrets.

## 1. Initialize and first commit (if not already done)

```bash
cd /path/to/ipl_intelligence_engine
git init
git add .
git status   # check what will be committed
git commit -m "Initial commit: IPL Intelligence Engine"
```

## 2. Create a repo on GitHub

- Go to [github.com/new](https://github.com/new).
- Name it e.g. `ipl-intelligence-engine`.
- Do **not** add a README, .gitignore, or license (they already exist locally).
- Create the repository.

## 3. Add remote and push

```bash
git remote add origin https://github.com/YOUR_USERNAME/ipl-intelligence-engine.git
git branch -M main
git push -u origin main
```

Use your GitHub username and repo name. For SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/ipl-intelligence-engine.git
git push -u origin main
```

## What is not uploaded

- **`.venv/`** — virtualenv; others use `pip install -r requirements-ui.txt` for the full local stack.
- **`models/saved/*.pkl`** — trained models; others run `python3 run_from_kaggle.py`.
- **`data/kaggle_ipl/`** — downloaded CSV; others run the pipeline to fetch from Kaggle.
- **`kaggle.json`** — never commit; each user adds their own.
