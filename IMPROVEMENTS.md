# How to Improve the IPL Intelligence Engine

Prioritised by impact and effort. Do the top items first for the biggest gain.

---

## 1. Use real ELO and form in the dashboard (high impact, medium effort)

**Problem:** The app uses neutral values (elo_delta=0, form_delta=0, h2h_venue_wr=0.5) for every match, so pre-match predictions don’t reflect team strength.

**Fix:**
- When you run the pipeline, **save the latest ELO and form** per team (and optionally H2H at venue) to a small JSON or pickle in `models/saved/`.
- In the dashboard, **look up** Team A and Team B in that file and set `pre_feats['elo_delta']`, `pre_feats['form_delta']`, `pre_feats['h2h_venue_wr']` from saved values (or defaults if team is new).
- For “future” matches, use the **most recent** saved ELO/form (e.g. end of last season or latest match in data).

**Result:** Pre-match accuracy should move toward the ~67% / 0.74 AUC you see in training when ELO and form are real.

---

## 2. Recalibrate pre-match probabilities (high impact, low effort)

**Problem:** Pre-match probabilities are poorly calibrated (e.g. when the model says 70%, the true rate is lower). Evaluation already flags “NEEDS RECALIBRATION”.

**Fix:**
- After training the pre-match LightGBM, fit **Platt scaling** (or isotonic regression) on the **validation set** (2023) to map raw probabilities to calibrated ones.
- Save the calibrator with the model and use it in the dashboard before showing win %.

**Result:** Stated win probabilities (e.g. 65%) will be closer to actual long-run frequencies.

---

## 3. Retrain when new data is available (ongoing)

**Problem:** Model is trained on data up to a point; new seasons or new matches are not reflected.

**Fix:**
- Re-run the full pipeline periodically (e.g. after each IPL season):  
  `python3 run_from_kaggle.py`
- Optionally trigger a **Kaggle dataset refresh** (re-download) so you get the latest IPL data before retraining.

**Result:** Predictions stay relevant for future and “recent past” matches.

---

## 4. Allow “replay” of a past match in the app (medium impact, medium effort)

**Problem:** You want to test “if I feed the real state of a past match, does it predict the winner?”

**Fix:**
- Add a **“Replay past match”** flow: user picks a match (e.g. from a dropdown of match_id or date + teams), app loads the **actual** over-by-over state (or key overs) from the dataset and shows how win probability and collapse risk would have evolved.
- Alternatively: add an option to **paste or upload one past match** (teams, venue, toss, 1st innings total, and 2nd innings score by over) and show the same curves.

**Result:** Easy way to validate the model on known outcomes and build trust.

---

## 5. Tune hyperparameters (medium impact, medium effort)

**Problem:** Current LightGBM and meta-learner settings are sensible but not optimised for your dataset.

**Fix:**
- Use **chronological** train/val splits (as now) and run a small **grid or random search** over:
  - Pre-match: `n_estimators`, `learning_rate`, `num_leaves`, `reg_alpha`, `reg_lambda`
  - In-match: same kind of range
  - Meta: `C` for LogisticRegression
- Optimise for **validation AUC** or **Brier score**; report test metrics only at the end.

**Result:** A few percent gain in AUC or accuracy is realistic.

---

## 6. Add more features (medium impact, higher effort)

**Ideas:**
- **Pre-match:** Day/night, rest days between matches, key player availability (if you have data), venue-specific team records.
- **In-match:** Over-by-over momentum (e.g. runs in last 2 overs), specific bowler/batter matchup if you have ball-by-ball player IDs.
- **Collapse:** Same momentum or recent wicket indicators.

**Result:** Richer signal can improve both pre-match and in-match models, especially where data is available.

---

## 7. Improve collapse definition or threshold (lower impact, low effort)

**Problem:** “3+ wickets in next 30 balls” is one possible definition; it might not match how you want to use “collapse risk”.

**Fix:**
- Try **2 wickets in 24 balls** or **3 in 36** and compare AUC / log-loss. Optionally add a **second model** for “2 wickets in 24” and show both in the app (e.g. “Quick wickets” vs “Full collapse”).
- Keep the current model as default; add the alternative as an option.

**Result:** Collapse alerts can better match your use case (e.g. T20 death overs).

---

## 8. Dashboard and UX

- **ELO/form override:** Let power users **manually** set “Team A strength” and “Team B strength” (or ELO delta) when they have a view (e.g. from another source).
- **Uncertainty:** Show a small “confidence” or “based on recent data through 2024” so users know the model’s scope.
- **Export:** “Download this prediction” (CSV or PDF) for record-keeping.

---

## 9. Code and robustness

- **Validation:** Check for missing columns and invalid ranges (e.g. score_2nd > score_1st + 1 when match is “complete”) and show a clear message in the app.
- **Tests:** Add a few unit tests for `build_pre_match_features`, `build_pressure_features`, and `predict_match` so refactors don’t break the pipeline.
- **Logging:** Log which model version and data range were used when saving artifacts (e.g. in `models/saved/version.txt` or inside the pickle metadata).

---

## Quick summary

| Priority | Improvement                    | Impact   | Effort   |
|----------|--------------------------------|----------|----------|
| 1        | Real ELO/form in dashboard     | High     | Medium   |
| 2        | Recalibrate pre-match probs    | High     | Low      |
| 3        | Retrain with new data          | High     | Low      |
| 4        | Replay past match in app       | Medium   | Medium   |
| 5        | Hyperparameter tuning          | Medium   | Medium   |
| 6        | More features                  | Medium   | Higher   |
| 7        | Collapse threshold/definition  | Lower    | Low      |
| 8        | Dashboard UX (override, export) | Medium   | Low–Med  |
| 9        | Validation, tests, logging     | Robustness | Medium |

Start with **1** and **2** for the biggest improvement in pre-match quality and trust in the numbers.
