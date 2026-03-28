# Get started — step by step

Follow these in order.

---

## Step 1: Get your Kaggle API key (one time)

1. Open **https://www.kaggle.com** and sign in.
2. Click your **profile picture** (top right) → **Settings**.
3. Scroll to **API** → click **Create New Token**.
4. A file **`kaggle.json`** will download (username + secret key).

---

## Step 2: Put the key where the project expects it

1. On your **Desktop**, create a folder named **`kaggle`** (if it’s not there already).
2. **Move** the downloaded **`kaggle.json`** into that folder.

You should have:

```
Desktop/
  kaggle/
    kaggle.json
```

The project is set up to use `Desktop/kaggle` automatically.

---

## Step 3: Install dependencies

Open **Terminal** and go to the project folder, then install:

```bash
cd /path/to/CricGyan
pip install -r requirements-ui.txt
```

---

## Step 4: Run the pipeline

From the **same** project folder, run:

```bash
python run_from_kaggle.py
```

This will:

1. Download the IPL dataset from Kaggle (using your `kaggle.json`).
2. Load and prepare the data.
3. Build features and train the models.
4. Print a short summary.

First run may take a few minutes (download + training).

---

## Step 5: Run the dashboard

From the project folder:

```bash
python3 -m streamlit run dashboard/app.py
```

Use `python3 -m streamlit` (not plain `streamlit`) so it works when the Streamlit executable isn’t on your PATH.

---

## If something goes wrong

- **“command not found: streamlit”**  
  Run: `python3 -m streamlit run dashboard/app.py` instead of `streamlit run dashboard/app.py`.

- **“Kaggle API key not found”**  
  Check that `kaggle.json` is inside `Desktop/kaggle/` and that the file is named exactly `kaggle.json`.

- **“Missing required columns”**  
  The Kaggle dataset might use different column names. Tell me the error message and we can fix the mapping.

- **You prefer using the notebook**  
  Open `notebooks/full_pipeline.ipynb` in Jupyter or VS Code, run the first cells (including “Option A: Load from Kaggle”), then run the rest in order.
