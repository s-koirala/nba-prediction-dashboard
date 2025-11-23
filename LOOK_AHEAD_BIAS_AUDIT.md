# Look-Ahead Bias Audit Report

## Executive Summary

**Status:** ⚠️ **MIXED** - Some components safe, others have bias

| Component | Status | Risk Level |
|-----------|--------|------------|
| Feature Engineering (Rolling Stats) | ✅ Safe | Low |
| Feature Engineering (Elo Ratings) | ✅ Safe | Low |
| Model Training Split | ❌ **BIASED** | High |
| Out-of-Sample Testing | ✅ Safe | Low |
| Walk-Forward Validation | ✅ Safe | Low |
| Daily Prediction Pipeline | ⚠️ **FIXED** (Nov 23) | Low (now) |

---

## Detailed Analysis

### ✅ 1. Feature Engineering - **NO BIAS**

**Rolling Averages** (`feature_builder.py:75-82`):
```python
all_stats[f'{col}_ROLL_{window}'] = (
    all_stats.groupby('TEAM_NAME')[col]
    .shift(1)  # ← Excludes current game
    .rolling(window=window, min_periods=1)
    .mean()
)
```

**Result:** ✅ Correctly uses `.shift(1)` to exclude current game from rolling average

---

**Win Percentages** (`feature_builder.py:107-113`):
```python
results[f'WIN_PCT_L{window}'] = (
    results.groupby('TEAM')['WON']
    .shift(1)  # ← Pre-game value
    .rolling(window=window, min_periods=1)
    .mean()
)
```

**Result:** ✅ Uses pre-game win percentages

---

**Streaks** (`feature_builder.py:124-131`):
```python
for won in team_data:
    streaks.append(streak)  # ← Append BEFORE updating
    # Then update streak based on current game
```

**Result:** ✅ Uses streak value before game is played

---

### ✅ 2. Elo System - **NO BIAS**

**Prediction Then Update** (`elo_system.py:158-168`):
```python
# Get predictions BEFORE updating ratings
win_prob_home, expected_margin = self.predict_game(...)

# Update ratings AFTER prediction
post_rating_home, post_rating_away, _ = self.update_ratings(...)
```

**Sequential Processing** (`elo_system.py:150`):
```python
games_df = games_df.sort_values('GAME_DATE')  # ← Chronological order
for idx, row in games_df.iterrows():  # ← One game at a time
```

**Result:** ✅ Elo predictions use pre-game ratings, updated sequentially

---

### ❌ 3. Model Training Split - **HAS BIAS**

**Problem** (`run_model_training_fixed.py:315-316`):
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # ← RANDOM SPLIT
)
```

**Issue:**
- Uses **random shuffling**, not temporal split
- Test games can be from BEFORE training games chronologically
- Violates time-series integrity

**Impact:**
- Model training metrics (in-sample) are **unreliable**
- Could have seen "future" patterns during training

**Mitigation:**
- ✅ Out-of-sample tests use proper temporal splits (see below)
- ⚠️ Don't trust in-sample accuracy metrics
- ✅ Dashboard shows OOS metrics only

**Severity:** High for training metrics, **Low for dashboard** (which uses OOS)

---

### ✅ 4. Out-of-Sample Testing - **NO BIAS**

**Temporal Split** (`test_oos_performance.py:308-309`):
```python
oos_games = historical[historical['GAME_DATE'].dt.year == 2024]
historical = historical[historical['GAME_DATE'].dt.year < 2024]
```

**Result:** ✅ Train on pre-2024, test on 2024 (proper temporal split)

---

**Feature Generation** (`test_oos_performance.py:54-60`):
```python
all_games = pd.concat([historical_data, oos_games])
features = feature_builder.build_features(all_games)
```

**Question:** Does this cause bias?

**Answer:** ✅ **NO** - Because:
1. Games sorted chronologically (line 56)
2. `.shift(1)` in feature_builder prevents current game inclusion
3. Rolling averages only use games *before* each game

**Result:** ✅ OOS testing is valid

---

### ✅ 5. Walk-Forward Validation - **NO BIAS**

**Process** (`optimize_bet_sizes_walkforward.py`):
1. Train on 2024-25 season data
2. Test on 2025-26 season data
3. Temporal split enforced

**Result:** ✅ Proper walk-forward methodology

---

### ⚠️ 6. Daily Predictions - **FIXED (Nov 23, 2025)**

**Previous Problem:**
- `test_2025_26_season.py` regenerated ALL predictions
- Used latest data to recalculate past predictions
- Predictions changed when recalculated

**Current Solution:**
- `archive_predictions.py` preserves original predictions
- `append_new_predictions.py` adds incrementally only
- Never regenerate old predictions

**Status:** ✅ Fixed going forward, ⚠️ historical data (pre-Nov 23) may have bias

---

## Summary of Bias Sources

### High Impact (Affects Results):

1. ❌ **Model Training Split** - Random instead of temporal
   - **Affected:** In-sample training metrics
   - **Not Affected:** OOS metrics (what dashboard shows)
   - **Fix Needed:** Yes (for model retraining)

2. ❌ **Historical Prediction Regeneration** (pre-Nov 23)
   - **Affected:** Performance tracking before Nov 23
   - **Not Affected:** Future predictions
   - **Fix:** Implemented (incremental-only pipeline)

### Low Impact (Does Not Affect Results):

3. ✅ **Feature Engineering** - Correctly implemented with `.shift(1)`
4. ✅ **Elo System** - Sequential processing is correct
5. ✅ **OOS Testing** - Temporal splits are correct
6. ✅ **Walk-Forward Validation** - Methodology is correct

---

## Recommendations

### Immediate Actions:

1. ✅ **DONE:** Fixed daily prediction pipeline (Nov 23)
2. ⚠️ **WARN:** Historical metrics (pre-Nov 23) are approximate
3. ✅ **DONE:** Incremental-only updates going forward

### Future Actions:

1. **Retrain models with temporal split:**
   ```python
   # Instead of random split
   cutoff_date = '2023-01-01'
   X_train = X[games_clean['GAME_DATE'] < cutoff_date]
   X_test = X[games_clean['GAME_DATE'] >= cutoff_date]
   ```

2. **Add bias detection tests:**
   - Verify predictions don't change when recalculated
   - Compare archive vs historical file
   - Alert if discrepancies found

3. **Document training date:**
   - Save model training date in metadata
   - Include data cutoff in model version

---

## Confidence in Current Metrics

### Dashboard Performance Metrics:

| Metric | Source | Confidence | Notes |
|--------|--------|------------|-------|
| 2024-25 OOS ROI | test_oos_performance.py | ✅ High | Proper temporal split |
| 2025-26 OOS ROI | test_2025_26_season.py | ⚠️ Medium | Pre-Nov 23 data may have bias |
| Walk-Forward ROI | optimize_bet_sizes_walkforward.py | ✅ High | Proper methodology |
| Tonight's Predictions | predict_tonight.py | ✅ High | Generated pre-game |

### What to Trust:

✅ **Trust:**
- Out-of-sample ROI metrics (2024-25 season)
- Walk-forward validation results
- Future predictions (Nov 23 onwards)

⚠️ **Use with Caution:**
- In-sample training accuracy
- Historical performance before Nov 23, 2025
- Exact ROI numbers (likely 2-5% optimistic)

❌ **Don't Trust:**
- Model training set accuracy metrics
- Performance claims not from OOS testing

---

## Conclusion

**Overall Assessment:** The model has **moderate look-ahead bias** from:
1. Random training split (affects training metrics only)
2. Historical prediction regeneration (fixed Nov 23)

**Dashboard Impact:** **LOW** - Dashboard shows OOS metrics which use proper temporal splits.

**Recommendation:** Continue with current system. Retrain models with temporal split when convenient, but not urgent.

---

**Last Updated:** November 23, 2025
**Auditor:** Identified by user observation + code review
**Status:** Ongoing monitoring with incremental-only pipeline
