"""
Serverless inference: numpy + lightgbm only (no pandas/sklearn/scipy).
Requires exported assets next to pickles — see models/inference_export.py.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np

from models.feature_constants import IN_MATCH_FEATURES, PRE_MATCH_FEATURES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVED = os.path.join(ROOT, "models", "saved")

_models: Optional[Tuple[Any, ...]] = None


def project_root() -> str:
    return ROOT


def _sigmoid_1d(coef: List[List[float]], intercept: List[float], x: float) -> float:
    z = coef[0][0] * x + intercept[0]
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def _sigmoid_2d(coef: List[List[float]], intercept: List[float], x1: float, x2: float) -> float:
    z = coef[0][0] * x1 + coef[0][1] * x2 + intercept[0]
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def _dict_to_row(feats: Dict[str, Any], keys: List[str]) -> np.ndarray:
    return np.array([[float(feats[k]) for k in keys]], dtype=np.float64)


def _build_toss_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for r in rows:
        era = r.get("era")
        if hasattr(era, "item"):
            era = str(era)
        else:
            era = str(era) if era is not None else ""
        key = (str(r.get("venue", "")), str(r.get("toss_decision", "")), era)
        te = r.get("toss_edge", 0.0)
        if te is None or (isinstance(te, float) and math.isnan(te)):
            te = 0.0
        out[key] = float(te)
    return out


def get_toss_edge_lite(
    toss_index: Dict[Tuple[str, str, str], float],
    venue: str,
    toss_decision: str,
    era: str,
) -> float:
    return float(toss_index.get((venue, toss_decision, era), 0.0))


def era_strip_data_lite(
    toss_index: Dict[Tuple[str, str, str], float],
    venue: str,
    toss_decision: str,
) -> List[Dict[str, Any]]:
    eras = [("early", "2008–12"), ("mid", "2013–18"), ("modern", "2019–25")]
    out: List[Dict[str, Any]] = []
    for era_key, label in eras:
        edge = get_toss_edge_lite(toss_index, venue, toss_decision, era_key)
        wr = (edge + 0.5) * 100
        out.append({"label": label, "edge_pct": round(edge * 100, 1), "win_rate_pct": round(wr, 1)})
    return out


def decay_note_lite(rows: List[Dict[str, Any]], venue: str, toss_decision: str) -> str:
    if not rows:
        return "Train the pipeline to load Bayesian toss-alpha table."
    sub = [r for r in rows if r.get("venue") == venue and r.get("toss_decision") == toss_decision]
    if not sub:
        return "No exact venue row — edges default toward neutral."
    adr = sub[0].get("alpha_decay_rate")
    if adr is None or (isinstance(adr, float) and math.isnan(adr)):
        return "Toss edge by era from historical matches (Beta posterior)."
    return (
        f"Alpha decay (early→modern): {float(adr):.2f}. Higher means toss mattered more in the past than now."
    )


def predict_match_lite(
    lgb_pre: lgb.Booster,
    lgb_in: lgb.Booster,
    cal: Dict[str, Any],
    pre_feats: Dict[str, Any],
    in_feats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    x_pre = _dict_to_row(pre_feats, PRE_MATCH_FEATURES)
    raw_pre_p = float(lgb_pre.predict(x_pre)[0])
    pre_cal = cal.get("pre_calibrator")
    if pre_cal is not None:
        pre_p = _sigmoid_1d(pre_cal["coef"], pre_cal["intercept"], raw_pre_p)
    else:
        pre_p = raw_pre_p

    if in_feats is None:
        return {
            "win_prob_team_a": round(pre_p, 3),
            "win_prob_team_b": round(1.0 - pre_p, 3),
            "source": "pre-match only",
        }

    x_in = _dict_to_row(in_feats, IN_MATCH_FEATURES)
    in_p = float(lgb_in.predict(x_in)[0])
    meta = cal["meta"]
    final = _sigmoid_2d(meta["coef"], meta["intercept"], pre_p, in_p)

    return {
        "win_prob_team_a": round(final, 3),
        "win_prob_team_b": round(1 - final, 3),
        "pre_match_signal": round(pre_p, 3),
        "in_match_signal": round(in_p, 3),
        "toss_contribution": round(float(pre_feats.get("toss_edge_score", 0.0)), 3),
        "source": "ensemble",
    }


def live_collapse_lite(
    runs_needed: int,
    balls_remaining: int,
    wickets_fallen: int,
    batter_sr_recent: float,
    partnership_balls: int,
    bowler_economy: float,
    collapse_model: lgb.Booster,
    collapse_cal: Dict[str, List],
) -> Dict[str, Any]:
    rrr = runs_needed * 6 / max(balls_remaining, 1)
    crr = max((120 - balls_remaining) * rrr / 6, 0.1)
    phase_pct = (120 - balls_remaining) / 120
    feats = {
        "rrr": rrr,
        "crr": crr,
        "rrr_gap": rrr - crr,
        "rrr_ratio": rrr / crr,
        "wickets_fallen": float(wickets_fallen),
        "wickets_remaining": float(10 - wickets_fallen),
        "phase_pct": phase_pct,
        "is_powerplay": float(int(phase_pct < 0.3)),
        "is_death": float(int(phase_pct > 0.8)),
        "batter_sr_recent": float(batter_sr_recent),
        "partnership_ball": float(partnership_balls),
        "bowler_economy_live": float(bowler_economy),
        "pressure_index": 0.0,
        "runs_needed": float(runs_needed),
        "balls_remaining": float(balls_remaining),
    }
    row = _dict_to_row(feats, IN_MATCH_FEATURES)
    raw = float(collapse_model.predict(row)[0])
    prob = _sigmoid_1d(collapse_cal["coef"], collapse_cal["intercept"], raw)
    tier = (
        "critical"
        if prob > 0.60
        else "high"
        if prob > 0.40
        else "moderate"
        if prob > 0.20
        else "low"
    )
    return {
        "collapse_probability": round(prob, 3),
        "risk_tier": tier,
        "rrr": round(rrr, 2),
        "rrr_gap": round(rrr - crr, 2),
    }


def load_models_cached():
    global _models
    if _models is not None:
        return _models
    try:
        lgb_pre = lgb.Booster(model_file=os.path.join(SAVED, "lgb_pre_model.txt"))
        lgb_in = lgb.Booster(model_file=os.path.join(SAVED, "lgb_in_model.txt"))
        with open(os.path.join(SAVED, "inference_calibrators.json"), "r", encoding="utf-8") as f:
            cal = json.load(f)
        with open(os.path.join(SAVED, "toss_alpha_table.json"), "r", encoding="utf-8") as f:
            toss_payload = json.load(f)
        toss_rows = toss_payload["rows"]
        toss_index = _build_toss_index(toss_rows)
        collapse_model = lgb.Booster(model_file=os.path.join(SAVED, "collapse_model.txt"))
        with open(os.path.join(SAVED, "team_stats.json"), "r", encoding="utf-8") as f:
            team_stats = json.load(f)
    except FileNotFoundError:
        _models = None
        return None
    _models = (lgb_pre, lgb_in, cal, toss_rows, toss_index, collapse_model, team_stats)
    return _models


def _stat(team_stats: Optional[Dict[str, Any]], team: str, key: str, default: float) -> float:
    if not team_stats:
        return default
    row = team_stats.get(team) or {}
    v = row.get(key, default)
    return float(v) if v is not None else default


def venue_tag(venue: str) -> str:
    return venue.split()[0].upper() if venue else ""


def run_predict(body: Dict[str, Any]) -> Dict[str, Any]:
    m = load_models_cached()
    if m is None or m[0] is None:
        raise RuntimeError("Models not found under models/saved/ (lite assets missing)")

    lgb_pre, lgb_in, cal, toss_rows, toss_index, collapse_model, team_stats = m

    team_a = body["team_a"]
    team_b = body["team_b"]
    venue = body["venue"]
    toss_winner_name = body["toss_winner"]
    toss_dec = body["toss_decision"]
    is_playoff = bool(body.get("playoff", False))

    current_over = int(body.get("over", 0))
    score_1st = int(body.get("score_1st", 0))
    score_2nd = int(body.get("score_2nd", 0))
    wkts_2nd = int(body.get("wkts_2nd", 0))
    batter_sr = float(body.get("batter_sr", 120))
    partnership = float(body.get("partnership", 6))
    bowler_eco = float(body.get("bowler_eco", 8.0))

    ERA_MODERN = "modern"
    toss_edge_raw = get_toss_edge_lite(toss_index, venue, toss_dec, ERA_MODERN)
    toss_edge_score = toss_edge_raw if toss_winner_name == team_a else -toss_edge_raw

    pre_feats = {
        "toss_edge_score": float(toss_edge_score),
        "elo_delta": float(_stat(team_stats, team_a, "elo", 1500) - _stat(team_stats, team_b, "elo", 1500)),
        "form_delta": float(_stat(team_stats, team_a, "form", 0.5) - _stat(team_stats, team_b, "form", 0.5)),
        "h2h_venue_wr": 0.5,
        "is_playoff": 1 if is_playoff else 0,
        "match_number_norm": 0.5,
    }

    use_live = current_over > 0 and score_1st > 0
    in_feats = None
    pressure_index = None
    rrr_gap = None
    crr = None
    rrr = None

    if use_live:
        runs_needed = max(0, score_1st + 1 - score_2nd)
        balls_remaining = max(1, (20 - current_over) * 6)
        balls_done = current_over * 6
        rrr = (runs_needed * 6) / balls_remaining
        crr = (score_2nd * 6) / balls_done if balls_done else rrr
        phase_pct = balls_done / 120.0
        rrr_gap_norm = max(0, min(1, (rrr - crr + 4) / 16))
        wkt_p = (wkts_2nd / 10) * (1 + phase_pct * 0.5)
        form_p = max(0, min(1, 1 - (batter_sr - 40) / 260)) if batter_sr <= 300 else 0
        part_p = 1 - min(partnership, 60) / 60
        bowl_n = 1 - (min(max(bowler_eco, 4), 14) - 4) / 10
        pressure_index = max(
            0,
            min(
                1,
                0.30 * rrr_gap_norm
                + 0.25 * wkt_p
                + 0.15 * phase_pct
                + 0.15 * form_p
                + 0.10 * part_p
                + 0.05 * bowl_n,
            ),
        )
        rrr_gap = rrr - crr
        in_feats = {
            "rrr": rrr,
            "crr": crr,
            "rrr_gap": rrr_gap,
            "rrr_ratio": rrr / crr if crr > 0 else 1.0,
            "wickets_fallen": wkts_2nd,
            "wickets_remaining": 10 - wkts_2nd,
            "phase_pct": phase_pct,
            "is_powerplay": 1 if current_over < 6 else 0,
            "is_death": 1 if current_over >= 16 else 0,
            "batter_sr_recent": float(batter_sr),
            "partnership_ball": partnership,
            "bowler_economy_live": bowler_eco,
            "pressure_index": pressure_index,
            "runs_needed": runs_needed,
            "balls_remaining": balls_remaining,
        }

    pred = predict_match_lite(lgb_pre, lgb_in, cal, pre_feats, in_feats)
    win_a = (pred.get("win_prob_team_a") or 0.5) * 100
    win_b = (pred.get("win_prob_team_b") or 0.5) * 100

    elo_d = pre_feats["elo_delta"]
    form_d = pre_feats["form_delta"]

    collapse_block: Dict[str, Any] = {"active": False}
    if use_live and collapse_model is not None and cal.get("collapse_cal"):
        runs_n = max(0, score_1st + 1 - score_2nd)
        balls_left = max(1, (20 - current_over) * 6)
        out = live_collapse_lite(
            runs_needed=runs_n,
            balls_remaining=balls_left,
            wickets_fallen=wkts_2nd,
            batter_sr_recent=float(batter_sr),
            partnership_balls=int(partnership),
            bowler_economy=bowler_eco,
            collapse_model=collapse_model,
            collapse_cal=cal["collapse_cal"],
        )
        prob = float(out.get("collapse_probability", 0))
        tier = str(out.get("risk_tier", "low")).upper()
        rrr_n = max(0, min(1, (rrr - crr + 4) / 16)) if use_live else 0
        wkt_p = (wkts_2nd / 10) * (1 + (current_over / 20) * 0.5) if use_live else 0
        ph_p = current_over / 20 if use_live else 0
        tot = max(1e-6, rrr_n + wkt_p + ph_p + 0.25)
        collapse_block = {
            "active": True,
            "tier": tier,
            "collapse_prob_pct": round(prob * 100, 1),
            "pressure_index": round(pressure_index or 0, 3),
            "bars": {
                "rrr": round(100 * rrr_n / tot) if tot else 0,
                "wkt": round(100 * wkt_p / tot) if tot else 0,
                "phase": round(100 * ph_p / tot) if tot else 0,
                "collapse": min(95, round(prob * 100)),
            },
        }

    conf = int(min(95, 42 + abs(win_a - 50) * 1.15))

    return {
        "win_a": round(win_a, 2),
        "win_b": round(win_b, 2),
        "confidence_pct": conf,
        "favourite": team_a if win_a >= win_b else team_b,
        "toss_edge_team_a": round(toss_edge_score, 4),
        "toss_modern_raw": round(toss_edge_raw, 4),
        "elo_delta": round(elo_d, 1),
        "form_delta": round(form_d, 3),
        "pred_source": pred.get("source", ""),
        "venue_tag": venue_tag(venue),
        "toss_tag_line": f"Edge (team A bat-first · {venue_tag(venue)} · modern): {toss_edge_score:+.3f}",
        "era_strip": era_strip_data_lite(toss_index, venue, toss_dec),
        "decay_note": decay_note_lite(toss_rows, venue, toss_dec),
        "h2h_placeholder": True,
        "rrr_gap": None if rrr_gap is None else round(rrr_gap, 2),
        "live": use_live,
        "pressure": collapse_block,
    }
