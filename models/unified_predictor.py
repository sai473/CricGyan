"""
Module 4: Unified Match Result Predictor
=========================================
Stacked ensemble:
  Layer 1a : LightGBM on pre-match features      → P_pre
  Layer 1b : LightGBM on in-match features       → P_in
  Meta      : Logistic regression on [P_pre, P_in] → final win probability

All splits are chronological (by season) — no data leakage.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from models.feature_constants import IN_MATCH_FEATURES, PRE_MATCH_FEATURES


# ── Pre-match model ───────────────────────────────────────────────────────────

def train_pre_match_model(pre_df: pd.DataFrame):
    train = pre_df[pre_df['year'] <= 2022]
    val   = pre_df[pre_df['year'] == 2023]
    test  = pre_df[pre_df['year'] >= 2024]

    X_tr, y_tr = train[PRE_MATCH_FEATURES], train['team_a_won']
    X_va, y_va = val[PRE_MATCH_FEATURES],   val['team_a_won']

    lgb_pre = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    lgb_pre.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Platt scaling on validation set for better-calibrated probabilities
    raw_va = lgb_pre.predict_proba(X_va)[:, 1]
    pre_calibrator = LogisticRegression(C=1.0)
    pre_calibrator.fit(raw_va.reshape(-1, 1), y_va)

    if len(test) > 0:
        raw_te = lgb_pre.predict_proba(test[PRE_MATCH_FEATURES])[:, 1]
        p = pre_calibrator.predict_proba(raw_te.reshape(-1, 1))[:, 1]
        print(f"Pre-match model — Test AUC: {roc_auc_score(test['team_a_won'], p):.4f}  "
              f"Brier: {brier_score_loss(test['team_a_won'], p):.4f}")
    return lgb_pre, pre_calibrator


# ── In-match model ────────────────────────────────────────────────────────────

def train_in_match_model(pressure_df: pd.DataFrame, pre_df: pd.DataFrame):
    """Uses over-boundary snapshots (ball_no % 6 == 0)."""
    snaps = pressure_df[pressure_df['ball_no'] % 6 == 0].copy()
    snaps = snaps.merge(pre_df[['match_id', 'team_a_won']], on='match_id', how='inner')

    train = snaps[snaps['season'] <= 2022]
    val   = snaps[snaps['season'] == 2023]
    test  = snaps[snaps['season'] >= 2024]

    X_tr, y_tr = train[IN_MATCH_FEATURES], train['team_a_won']
    X_va, y_va = val[IN_MATCH_FEATURES],   val['team_a_won']

    lgb_in = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.5,
        random_state=42, verbose=-1,
    )
    lgb_in.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    if len(test) > 0:
        p = lgb_in.predict_proba(test[IN_MATCH_FEATURES])[:, 1]
        print(f"In-match model  — Test AUC: {roc_auc_score(test['team_a_won'], p):.4f}  "
              f"Brier: {brier_score_loss(test['team_a_won'], p):.4f}")
    return lgb_in


# ── Meta-learner ──────────────────────────────────────────────────────────────

def train_meta_learner(lgb_pre, lgb_in, pressure_df: pd.DataFrame, pre_df: pd.DataFrame, pre_calibrator=None):
    """
    Trains on over-10 snapshots with OOF pre-match probabilities to avoid leakage.
    If pre_calibrator is provided, pre-match probs are calibrated so meta sees same scale as at inference.
    """
    snap10 = pressure_df[pressure_df['over'] == 10].copy()
    snap10 = snap10.merge(pre_df[['match_id', 'team_a_won'] + PRE_MATCH_FEATURES],
                          on='match_id', how='inner')

    train10 = snap10[snap10['season'] <= 2022]

    raw_pre = lgb_pre.predict_proba(train10[PRE_MATCH_FEATURES])[:, 1]
    pre_prob_tr = pre_calibrator.predict_proba(raw_pre.reshape(-1, 1))[:, 1] if pre_calibrator is not None else raw_pre
    in_prob_tr  = lgb_in.predict_proba(train10[IN_MATCH_FEATURES])[:, 1]

    meta = LogisticRegression(C=1.0)
    meta.fit(np.column_stack([pre_prob_tr, in_prob_tr]), train10['team_a_won'])
    print(f"Meta-learner weights: pre={meta.coef_[0][0]:.3f}, in={meta.coef_[0][1]:.3f}")
    return meta


# ── Full training pipeline ────────────────────────────────────────────────────

def train_all(pre_df: pd.DataFrame, pressure_df: pd.DataFrame):
    """Train all three layers and return them."""
    print("\n[1/3] Training pre-match model ...")
    lgb_pre, pre_calibrator = train_pre_match_model(pre_df)

    print("\n[2/3] Training in-match model ...")
    lgb_in  = train_in_match_model(pressure_df, pre_df)

    print("\n[3/3] Training meta-learner ...")
    meta    = train_meta_learner(lgb_pre, lgb_in, pressure_df, pre_df, pre_calibrator)

    print("\nAll models trained.")
    return lgb_pre, lgb_in, meta, pre_calibrator


# ── Live win probability curve ────────────────────────────────────────────────

def live_win_curve(
    match_id: str,
    pressure_df: pd.DataFrame,
    pre_feats: dict,
    lgb_pre, lgb_in, meta,
) -> pd.DataFrame:
    """
    Returns a DataFrame with P(team A wins) at every over boundary.
    pre_feats: dict matching PRE_MATCH_FEATURES keys.
    """
    balls = pressure_df[pressure_df['match_id'] == match_id].copy()
    pre_p = lgb_pre.predict_proba(pd.DataFrame([pre_feats]))[0, 1]
    curve = []

    for ov in range(1, 21):
        snap = balls[balls['over'] == ov - 1].tail(1)
        if snap.empty:
            continue
        in_p   = lgb_in.predict_proba(snap[IN_MATCH_FEATURES])[0, 1]
        final  = meta.predict_proba([[pre_p, in_p]])[0, 1]
        curve.append({
            'over':           ov,
            'win_prob_team_a': round(final, 3),
            'win_prob_team_b': round(1 - final, 3),
            'pressure_index':  round(float(snap['pressure_index'].iloc[0]), 3),
            'rrr':             round(float(snap['rrr'].iloc[0]), 2),
        })

    return pd.DataFrame(curve)


# ── Single-match prediction ───────────────────────────────────────────────────

def predict_match(
    lgb_pre, lgb_in, meta,
    pre_feats: dict,
    in_feats: dict = None,
    pre_calibrator=None,
) -> dict:
    """
    pre_feats : dict with PRE_MATCH_FEATURES keys
    in_feats  : dict with IN_MATCH_FEATURES keys (None = pre-match only)
    pre_calibrator : optional LogisticRegression to calibrate pre-match probability
    """
    raw_pre_p = lgb_pre.predict_proba(pd.DataFrame([pre_feats]))[0, 1]
    pre_p = pre_calibrator.predict_proba([[raw_pre_p]])[0, 1] if pre_calibrator is not None else raw_pre_p

    if in_feats is None:
        return {
            'win_prob_team_a': round(pre_p, 3),
            'win_prob_team_b': round(1.0 - pre_p, 3),
            'source': 'pre-match only',
        }

    in_p   = lgb_in.predict_proba(pd.DataFrame([in_feats]))[0, 1]
    final  = meta.predict_proba([[pre_p, in_p]])[0, 1]

    return {
        'win_prob_team_a':  round(final, 3),
        'win_prob_team_b':  round(1 - final, 3),
        'pre_match_signal': round(pre_p, 3),
        'in_match_signal':  round(in_p, 3),
        'toss_contribution': round(pre_feats.get('toss_edge_score', 0.0), 3),
        'source': 'ensemble',
    }
