"""
Module 1: Ball-by-ball Pressure Index + Batting Collapse Predictor
==================================================================
Inputs : raw IPL ball-by-ball DataFrame
Outputs: feature-engineered DataFrame with pressure_index and collapse_next_30 label
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from models.feature_constants import COLLAPSE_FEATURES


# ── Feature engineering ───────────────────────────────────────────────────────

def build_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Works on 2nd-innings chases only.
    Returns one row per ball with pressure signals + collapse label.
    """
    chases = df[
        (df['innings'] == 2) &
        (df['runs_target'].notna())
    ].copy().sort_values(['match_id', 'ball_no'])

    # ── Core pressure signals ─────────────────────────────────────────────
    chases['runs_scored_so_far'] = chases.groupby('match_id')['runs_total'].cumsum()
    chases['runs_needed']        = chases['runs_target'] - chases['runs_scored_so_far']
    chases['balls_remaining']    = (120 - chases['ball_no']).clip(lower=1)

    chases['rrr']       = (chases['runs_needed'] * 6) / chases['balls_remaining']
    chases['crr']       = (chases['runs_scored_so_far'] * 6) / chases['ball_no'].clip(lower=1)
    chases['rrr_gap']   = chases['rrr'] - chases['crr']
    chases['rrr_ratio'] = chases['rrr'] / chases['crr'].clip(lower=0.1)

    chases['wickets_fallen']   = chases.groupby('match_id')['wicket_kind'].transform(
        lambda x: x.notna().cumsum()
    )
    chases['wickets_remaining'] = 10 - chases['wickets_fallen']
    chases['phase_pct']         = chases['ball_no'] / 120
    chases['is_powerplay']      = (chases['over'] < 6).astype(int)
    chases['is_death']          = (chases['over'] >= 16).astype(int)

    # ── Rolling batter strike rate (last 12 balls) ────────────────────────
    def rolling_batter_sr(grp, window=12):
        grp = grp.copy()
        grp['batter_runs_roll']  = grp['runs_batter'].rolling(window, min_periods=1).sum()
        grp['batter_balls_roll'] = grp['valid_ball'].rolling(window, min_periods=1).sum()
        grp['batter_sr_recent']  = grp['batter_runs_roll'] / grp['batter_balls_roll'].clip(lower=1) * 100
        return grp

    chases = chases.groupby(['match_id', 'batter'], group_keys=False).apply(rolling_batter_sr)

    # ── Partnership age ───────────────────────────────────────────────────
    def partnership_balls(grp):
        grp = grp.sort_values('ball_no').copy()
        grp['wicket_event']   = grp['wicket_kind'].notna().cumsum()
        grp['partnership_ball'] = grp.groupby('wicket_event').cumcount()
        return grp

    chases = chases.groupby('match_id', group_keys=False).apply(partnership_balls)

    # ── Live bowler economy ───────────────────────────────────────────────
    def bowler_economy_live(grp):
        grp = grp.copy()
        grp['bowler_runs_cum']  = grp.groupby('bowler')['runs_total'].transform(
            lambda x: x.expanding().sum().shift(1).fillna(0))
        grp['bowler_balls_cum'] = grp.groupby('bowler')['valid_ball'].transform(
            lambda x: x.expanding().sum().shift(1).fillna(0))
        grp['bowler_economy_live'] = (
            grp['bowler_runs_cum'] / grp['bowler_balls_cum'].clip(lower=1) * 6
        )
        return grp

    chases = chases.groupby('match_id', group_keys=False).apply(bowler_economy_live)

    # ── Composite pressure index (interpretable weighted sum) ─────────────
    rrr_gap_norm  = chases['rrr_gap'].clip(-4, 12).sub(-4).div(16)
    wkt_pressure  = (chases['wickets_fallen'] / 10 * (1 + chases['phase_pct'] * 0.5)).clip(0, 1)
    form_pressure = (1 - (chases['batter_sr_recent'] - 40).clip(0, 260) / 260)
    part_pressure = (1 - chases['partnership_ball'].clip(0, 60) / 60)
    bowl_norm     = (chases['bowler_economy_live'].clip(4, 14).sub(4).div(10).rsub(1))

    chases['pressure_index'] = (
        0.30 * rrr_gap_norm +
        0.25 * wkt_pressure +
        0.15 * chases['phase_pct'] +
        0.15 * form_pressure +
        0.10 * part_pressure +
        0.05 * bowl_norm
    ).clip(0, 1)

    # ── Collapse label: 3+ wickets in next 30 balls ───────────────────────
    def label_collapse(grp):
        grp = grp.sort_values('ball_no').copy()
        future_wkts = (
            grp['wicket_kind'].notna()
              .iloc[::-1]
              .rolling(30, min_periods=1)
              .sum()
              .iloc[::-1]
        )
        grp['collapse_next_30'] = (future_wkts >= 3).astype(int)
        return grp

    chases = chases.groupby('match_id', group_keys=False).apply(label_collapse)

    keep = [
        'match_id', 'ball_no', 'over', 'batter', 'bowler', 'season',
        'rrr', 'crr', 'rrr_gap', 'rrr_ratio',
        'wickets_fallen', 'wickets_remaining',
        'phase_pct', 'is_powerplay', 'is_death',
        'batter_sr_recent', 'batter_runs_roll',
        'partnership_ball', 'bowler_economy_live',
        'pressure_index', 'runs_needed', 'balls_remaining',
        'venue', 'collapse_next_30',
    ]
    return chases[[c for c in keep if c in chases.columns]]


# ── Model training ────────────────────────────────────────────────────────────


def train_collapse_model(features_df: pd.DataFrame):
    """
    Train on seasons ≤2022, validate on 2023, test on 2024+.
    Returns (lgbm_model, calibrator, predict_fn).
    """
    df = features_df.dropna(subset=COLLAPSE_FEATURES + ['collapse_next_30'])

    train = df[df['season'] <= 2022]
    val   = df[df['season'] == 2023]
    test  = df[df['season'] >= 2024]

    X_tr, y_tr = train[COLLAPSE_FEATURES], train['collapse_next_30']
    X_va, y_va = val[COLLAPSE_FEATURES],   val['collapse_next_30']
    X_te, y_te = test[COLLAPSE_FEATURES],  test['collapse_next_30']

    params = dict(
        objective='binary', metric=['binary_logloss', 'auc'],
        learning_rate=0.03, num_leaves=63, max_depth=6,
        min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, lambda_l1=0.1, lambda_l2=1.0,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        verbose=-1,
    )

    model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X_va, y_va)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    # Platt calibration
    cal = LogisticRegression()
    cal.fit(model.predict(X_va).reshape(-1, 1), y_va)

    def predict(X):
        raw = model.predict(X)
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]

    if len(X_te) > 0:
        te_prob = predict(X_te)
        print(f"Collapse model — Test AUC: {roc_auc_score(y_te, te_prob):.4f}  "
              f"Log-loss: {log_loss(y_te, te_prob):.4f}")

    return model, cal, predict


# ── Live inference ────────────────────────────────────────────────────────────

def live_collapse_probability(
    runs_needed: int,
    balls_remaining: int,
    wickets_fallen: int,
    batter_sr_recent: float,
    partnership_balls: int,
    bowler_economy: float,
    model,
    calibrator,
) -> dict:
    """Call after every ball to get a live collapse risk reading."""
    rrr       = runs_needed * 6 / max(balls_remaining, 1)
    crr       = max((120 - balls_remaining) * rrr / 6, 0.1)
    phase_pct = (120 - balls_remaining) / 120

    row = pd.DataFrame([{
        'rrr': rrr, 'crr': crr, 'rrr_gap': rrr - crr,
        'rrr_ratio': rrr / crr,
        'wickets_fallen': wickets_fallen,
        'wickets_remaining': 10 - wickets_fallen,
        'phase_pct': phase_pct,
        'is_powerplay': int(phase_pct < 0.3),
        'is_death': int(phase_pct > 0.8),
        'batter_sr_recent': batter_sr_recent,
        'partnership_ball': partnership_balls,
        'bowler_economy_live': bowler_economy,
        'pressure_index': 0.0,
        'runs_needed': runs_needed,
        'balls_remaining': balls_remaining,
    }])

    raw  = model.predict(row)[0]
    prob = calibrator.predict_proba([[raw]])[0][1]
    tier = ('critical' if prob > 0.60 else
            'high'     if prob > 0.40 else
            'moderate' if prob > 0.20 else 'low')

    return {'collapse_probability': round(prob, 3), 'risk_tier': tier,
            'rrr': round(rrr, 2), 'rrr_gap': round(rrr - crr, 2)}
