"""
Export native LightGBM model files + JSON sidecars for serverless inference
(numpy + lightgbm only — no pandas/sklearn/scipy at runtime).

Call from training (`run_from_kaggle.py`) or standalone after pickles exist:
  python -m models.inference_export
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

import numpy as np


def _lr_predict_proba_1d(coef: list, intercept: list, x: float) -> float:
    """Match sklearn LogisticRegression binary predict_proba positive class (1 feature)."""
    z = coef[0][0] * x + intercept[0]
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def _lr_predict_proba_2d(coef: list, intercept: list, x1: float, x2: float) -> float:
    z = coef[0][0] * x1 + coef[0][1] * x2 + intercept[0]
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def export_lite_assets(
    save_dir: str,
    lgb_pre: Any,
    lgb_in: Any,
    meta: Any,
    pre_calibrator: Any,
    collapse_model: Any,
    collapse_cal: Any,
    toss_alpha_df: Any,
    team_stats: Dict[str, Any],
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    lgb_pre.booster_.save_model(os.path.join(save_dir, "lgb_pre_model.txt"))
    lgb_in.booster_.save_model(os.path.join(save_dir, "lgb_in_model.txt"))
    collapse_model.save_model(os.path.join(save_dir, "collapse_model.txt"))

    cal: Dict[str, Any] = {
        "meta": {
            "coef": meta.coef_.tolist(),
            "intercept": meta.intercept_.tolist(),
        },
        "pre_calibrator": None,
        "collapse_cal": {
            "coef": collapse_cal.coef_.tolist(),
            "intercept": collapse_cal.intercept_.tolist(),
        },
    }
    if pre_calibrator is not None:
        cal["pre_calibrator"] = {
            "coef": pre_calibrator.coef_.tolist(),
            "intercept": pre_calibrator.intercept_.tolist(),
        }

    with open(os.path.join(save_dir, "inference_calibrators.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=0)

    tdf = toss_alpha_df.copy()
    if "era" in tdf.columns:
        tdf["era"] = tdf["era"].astype(str)
    rows = tdf.astype(object).where(tdf.notna(), None).to_dict(orient="records")

    with open(os.path.join(save_dir, "toss_alpha_table.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f)

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    with open(os.path.join(save_dir, "team_stats.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(team_stats), f)

    # Sanity: lite math matches sklearn on a probe
    raw_pre = 0.42
    if pre_calibrator is not None and cal["pre_calibrator"] is not None:
        sk = float(pre_calibrator.predict_proba([[raw_pre]])[0, 1])
        lx = _lr_predict_proba_1d(cal["pre_calibrator"]["coef"], cal["pre_calibrator"]["intercept"], raw_pre)
        if abs(sk - lx) > 1e-4:
            raise RuntimeError(f"pre_calibrator mismatch sklearn={sk} lite={lx}")

    sk_m = float(meta.predict_proba([[0.52, 0.48]])[0, 1])
    lx_m = _lr_predict_proba_2d(cal["meta"]["coef"], cal["meta"]["intercept"], 0.52, 0.48)
    if abs(sk_m - lx_m) > 1e-4:
        raise RuntimeError(f"meta mismatch sklearn={sk_m} lite={lx_m}")

    raw_c = 0.35
    sk_c = float(collapse_cal.predict_proba([[raw_c]])[0, 1])
    lx_c = _lr_predict_proba_1d(cal["collapse_cal"]["coef"], cal["collapse_cal"]["intercept"], raw_c)
    if abs(sk_c - lx_c) > 1e-4:
        raise RuntimeError(f"collapse_cal mismatch sklearn={sk_c} lite={lx_c}")

    from models.feature_constants import IN_MATCH_FEATURES, PRE_MATCH_FEATURES

    x_pre = np.zeros((1, len(PRE_MATCH_FEATURES)))
    sk_pp = float(lgb_pre.predict_proba(x_pre)[0, 1])
    bo_pp = float(lgb_pre.booster_.predict(x_pre)[0])
    if abs(sk_pp - bo_pp) > 1e-4:
        raise RuntimeError(f"lgb_pre booster vs LGBMClassifier.predict_proba mismatch {bo_pp} vs {sk_pp}")

    x_in = np.zeros((1, len(IN_MATCH_FEATURES)))
    sk_ip = float(lgb_in.predict_proba(x_in)[0, 1])
    bo_ip = float(lgb_in.booster_.predict(x_in)[0])
    if abs(sk_ip - bo_ip) > 1e-4:
        raise RuntimeError(f"lgb_in booster vs LGBMClassifier.predict_proba mismatch {bo_ip} vs {sk_ip}")

    print("  Exported lite serverless assets: lgb_*_model.txt, inference_calibrators.json, toss_alpha_table.json, team_stats.json")


def export_lite_from_pickles(save_dir: str) -> None:
    """One-off: load existing *.pkl and write lite assets (needs full training env)."""
    import pickle

    with open(os.path.join(save_dir, "lgb_pre.pkl"), "rb") as f:
        lgb_pre = pickle.load(f)
    with open(os.path.join(save_dir, "lgb_in.pkl"), "rb") as f:
        lgb_in = pickle.load(f)
    with open(os.path.join(save_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(save_dir, "collapse_model.pkl"), "rb") as f:
        collapse_model = pickle.load(f)
    with open(os.path.join(save_dir, "collapse_cal.pkl"), "rb") as f:
        collapse_cal = pickle.load(f)
    with open(os.path.join(save_dir, "toss_alpha.pkl"), "rb") as f:
        toss_alpha_df = pickle.load(f)
    pre_calibrator = None
    try:
        with open(os.path.join(save_dir, "pre_calibrator.pkl"), "rb") as f:
            pre_calibrator = pickle.load(f)
    except FileNotFoundError:
        pass
    with open(os.path.join(save_dir, "team_stats.pkl"), "rb") as f:
        team_stats = pickle.load(f)

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


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "saved")
    export_lite_from_pickles(os.path.abspath(d))
