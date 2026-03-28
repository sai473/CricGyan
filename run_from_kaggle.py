#!/usr/bin/env python3
"""
Run the full IPL pipeline using data from Kaggle.

From the project folder, run:
  python run_from_kaggle.py

Before first run:
  1. Get kaggle.json from https://www.kaggle.com/settings (Create New Token)
  2. Put it in Desktop/kaggle/kaggle.json
  3. pip install -r requirements-ui.txt
"""

import os
import sys
import pandas as pd

# Run from project root so imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    print("=" * 60)
    print("  IPL Intelligence Engine — run from Kaggle")
    print("=" * 60)

    # 1. Load data from Kaggle
    print("\n[1/7] Loading data from Kaggle ...")
    from data.kaggle_ipl import load_ipl_from_kaggle
    df = load_ipl_from_kaggle(download_if_missing=True)
    print(f"      Loaded {len(df):,} balls, {df['match_id'].nunique():,} matches.")

    # 2. Toss alpha decay
    print("\n[2/7] Toss alpha decay ...")
    from models.toss_alpha_decay import compute_toss_alpha_decay, print_decay_report
    toss_alpha_df = compute_toss_alpha_decay(df)
    print_decay_report(toss_alpha_df)

    # 3. Pre-match features
    print("\n[3/7] Building pre-match features ...")
    from models.pre_match_features import build_pre_match_features
    pre_df = build_pre_match_features(df)
    print(f"      Pre-match rows: {len(pre_df)}")

    # 4. Pressure features
    print("\n[4/7] Building pressure (chase) features ...")
    from models.pressure_index import build_pressure_features
    pressure_df = build_pressure_features(df)
    print(f"      Pressure rows: {len(pressure_df)}")

    # 5. Train collapse predictor
    print("\n[5/7] Training collapse predictor ...")
    from models.pressure_index import train_collapse_model
    collapse_model, collapse_cal, collapse_predict = train_collapse_model(pressure_df)

    # 6. Train unified match predictor
    print("\n[6/7] Training unified match predictor ...")
    from models.unified_predictor import train_all
    lgb_pre, lgb_in, meta, pre_calibrator = train_all(pre_df, pressure_df)

    # 7. Evaluate
    print("\n[7/7] Running evaluation ...")
    from evaluation.evaluate import run_full_evaluation
    test_pre = pre_df[pre_df["year"] >= 2023]
    test_in = pressure_df[pressure_df["season"] >= 2023]
    run_full_evaluation(lgb_pre, lgb_in, meta, test_pre, test_in)

    # Save models and team stats for dashboard
    save_dir = os.path.join(PROJECT_ROOT, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)
    import pickle
    for name, obj in [
        ("lgb_pre", lgb_pre),
        ("lgb_in", lgb_in),
        ("meta", meta),
        ("pre_calibrator", pre_calibrator),
        ("collapse_model", collapse_model),
        ("collapse_cal", collapse_cal),
        ("toss_alpha", toss_alpha_df),
    ]:
        path = os.path.join(save_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {name}.pkl")

    # Latest ELO and form per team (for dashboard pre-match predictions)
    ord_col = pre_df["event_match_no"] if "event_match_no" in pre_df.columns else pd.Series(range(len(pre_df)), index=pre_df.index)
    pre_df["_ord"] = pre_df["year"].astype(float) * 1000 + ord_col.astype(float)
    bat = pre_df[["_ord", "batting_team", "elo_team_a", "form_a"]].rename(columns={"batting_team": "team", "elo_team_a": "elo", "form_a": "form"})
    bowl = pre_df[["_ord", "bowling_team", "elo_team_b", "form_b"]].rename(columns={"bowling_team": "team", "elo_team_b": "elo", "form_b": "form"})
    both = pd.concat([bat, bowl], ignore_index=True)
    latest = both.loc[both.groupby("team")["_ord"].idxmax()][["team", "elo", "form"]]
    team_stats = latest.set_index("team").to_dict("index")
    with open(os.path.join(save_dir, "team_stats.pkl"), "wb") as f:
        pickle.dump(team_stats, f)
    print("  Saved team_stats.pkl (latest ELO & form per team)")

    from models.inference_export import export_lite_assets

    export_lite_assets(
        save_dir,
        lgb_pre,
        lgb_in,
        meta,
        pre_calibrator,
        collapse_model,
        collapse_cal,
        toss_alpha_df,
        team_stats,
    )

    print("\n" + "=" * 60)
    print("  Done. To start the dashboard: python3 -m streamlit run dashboard/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
