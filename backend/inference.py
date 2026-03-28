"""
Shared inference for the web UI — same logic as dashboard/app.py.

If `models/saved/` contains lite export files (native LightGBM + JSON), uses
`inference_lite` (numpy + lightgbm only — fits Vercel Lambda ~500 MB). Otherwise
loads sklearn pickles + pandas (local / Streamlit).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVED = os.path.join(ROOT, "models", "saved")


def _lite_assets_ready() -> bool:
    names = (
        "lgb_pre_model.txt",
        "lgb_in_model.txt",
        "collapse_model.txt",
        "inference_calibrators.json",
        "toss_alpha_table.json",
        "team_stats.json",
    )
    return all(os.path.isfile(os.path.join(SAVED, n)) for n in names)


if _lite_assets_ready():
    from backend.inference_lite import load_models_cached, project_root, run_predict
else:
    import pickle

    import pandas as pd

    _models: Optional[tuple] = None

    def project_root() -> str:
        return ROOT

    def load_models_cached():
        global _models
        if _models is not None:
            return _models
        try:
            with open(os.path.join(SAVED, "lgb_pre.pkl"), "rb") as f:
                lgb_pre = pickle.load(f)
            with open(os.path.join(SAVED, "lgb_in.pkl"), "rb") as f:
                lgb_in = pickle.load(f)
            with open(os.path.join(SAVED, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)
            with open(os.path.join(SAVED, "toss_alpha.pkl"), "rb") as f:
                toss_df = pickle.load(f)
            with open(os.path.join(SAVED, "collapse_model.pkl"), "rb") as f:
                collapse_model = pickle.load(f)
            with open(os.path.join(SAVED, "collapse_cal.pkl"), "rb") as f:
                collapse_cal = pickle.load(f)
        except FileNotFoundError:
            _models = None
            return None
        pre_calibrator = team_stats = None
        try:
            with open(os.path.join(SAVED, "pre_calibrator.pkl"), "rb") as f:
                pre_calibrator = pickle.load(f)
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(SAVED, "team_stats.pkl"), "rb") as f:
                team_stats = pickle.load(f)
        except FileNotFoundError:
            pass
        _models = (lgb_pre, lgb_in, meta, toss_df, collapse_model, collapse_cal, pre_calibrator, team_stats)
        return _models

    def _stat(team_stats, team: str, key: str, default):
        return (team_stats or {}).get(team, {}).get(key, default)

    def venue_tag(venue: str) -> str:
        return venue.split()[0].upper() if venue else ""

    def era_strip_data(toss_df, venue: str, toss_decision: str) -> List[Dict[str, Any]]:
        from models.toss_alpha_decay import get_toss_edge_feature

        eras = [("early", "2008–12"), ("mid", "2013–18"), ("modern", "2019–25")]
        out = []
        for era_key, label in eras:
            edge = (
                get_toss_edge_feature(toss_df, venue, toss_decision, era_key)
                if toss_df is not None
                else 0.0
            )
            wr = (edge + 0.5) * 100
            out.append({"label": label, "edge_pct": round(edge * 100, 1), "win_rate_pct": round(wr, 1)})
        return out

    def decay_note_text(toss_df, venue: str, toss_decision: str) -> str:
        if toss_df is None or not len(toss_df):
            return "Train the pipeline to load Bayesian toss-alpha table."
        sub = toss_df[(toss_df["venue"] == venue) & (toss_df["toss_decision"] == toss_decision)]
        if sub.empty:
            return "No exact venue row — edges default toward neutral."
        row = sub.iloc[0]
        adr = row.get("alpha_decay_rate", float("nan"))
        if pd.isna(adr):
            return "Toss edge by era from historical matches (Beta posterior)."
        return f"Alpha decay (early→modern): {float(adr):.2f}. Higher means toss mattered more in the past than now."

    def run_predict(body: Dict[str, Any]) -> Dict[str, Any]:
        from models.pressure_index import live_collapse_probability
        from models.toss_alpha_decay import get_toss_edge_feature
        from models.unified_predictor import predict_match

        m = load_models_cached()
        if m is None or m[0] is None:
            raise RuntimeError("Models not found under models/saved/")

        lgb_pre, lgb_in, meta, toss_df, collapse_model, collapse_cal, pre_calibrator, team_stats = m

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
        toss_edge_raw = (
            get_toss_edge_feature(toss_df, venue, toss_dec, ERA_MODERN) if toss_df is not None else 0.0
        )
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

        if use_live and lgb_in is not None:
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

        pred = predict_match(lgb_pre, lgb_in, meta, pre_feats, in_feats, pre_calibrator=pre_calibrator)
        win_a = (pred.get("win_prob_team_a") or 0.5) * 100
        win_b = (pred.get("win_prob_team_b") or 0.5) * 100

        elo_d = pre_feats["elo_delta"]
        form_d = pre_feats["form_delta"]

        collapse_block: Dict[str, Any] = {"active": False}
        if use_live and collapse_model is not None and collapse_cal is not None:
            runs_n = max(0, score_1st + 1 - score_2nd)
            balls_left = max(1, (20 - current_over) * 6)
            out = live_collapse_probability(
                runs_needed=runs_n,
                balls_remaining=balls_left,
                wickets_fallen=wkts_2nd,
                batter_sr_recent=float(batter_sr),
                partnership_balls=int(partnership),
                bowler_economy=bowler_eco,
                model=collapse_model,
                calibrator=collapse_cal,
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
            "era_strip": era_strip_data(toss_df, venue, toss_dec),
            "decay_note": decay_note_text(toss_df, venue, toss_dec),
            "h2h_placeholder": True,
            "rrr_gap": None if rrr_gap is None else round(rrr_gap, 2),
            "live": use_live,
            "pressure": collapse_block,
        }
