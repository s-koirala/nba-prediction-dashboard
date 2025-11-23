# Prediction Integrity & Look-Ahead Bias Prevention

## The Problem (Discovered Nov 23, 2025)

**Look-ahead bias** was identified in the historical performance data:

### What Happened:
1. Original script (`test_2025_26_season.py`) **regenerated all predictions** using the latest data
2. Features (rolling averages, Elo ratings) were recalculated with information that wasn't available before games
3. Predictions for past games changed when recalculated with newer data
4. **Result:** Reported accuracy/ROI was inflated and unreliable

### Example:
- **Nov 22 pre-game predictions:** Nuggets, Bucks, Bulls, Pelicans, Magic, Mavericks to cover
- **Nov 23 regenerated predictions:** Only Nuggets, Bulls, Magic showed as predicted to cover
- **Why:** Rolling stats for Nov 22 games included data from Nov 23 games

This is a critical data integrity issue that makes backtesting results unreliable.

---

## The Solution

### **Incremental-Only Predictions**

Going forward, predictions are:
1. ✅ **Generated BEFORE games** (`predict_tonight.py` at 9 AM EST)
2. ✅ **Archived daily** (`archive_predictions.py` saves original predictions)
3. ✅ **Appended incrementally** (`append_new_predictions.py` adds completed games)
4. ❌ **Never regenerated** (old predictions are frozen)

### **Files & Purpose:**

| File | Purpose | When Used |
|------|---------|-----------|
| `results/tonights_predictions.csv` | Today's games (pre-game) | Displayed on dashboard |
| `results/prediction_archive/predictions_YYYY-MM-DD.csv` | Daily archives | Preserved originals |
| `results/predictions_2025_26.csv` | Historical performance | Incremental appends only |

### **Workflows:**

**Daily (9 AM EST):**
```bash
1. Archive yesterday's predictions
2. Generate tonight's predictions
3. Commit both to git
```

**Weekly (Sundays):**
```bash
1. Append completed games to historical file
2. Use archived predictions (not regenerated)
3. Commit updated historical data
```

---

## Impact on Historical Data

### **Data Before Nov 23, 2025:**
- ⚠️ May contain look-ahead bias (regenerated multiple times)
- Use with caution for accuracy claims
- Consider as "approximate" performance

### **Data After Nov 23, 2025:**
- ✅ Original predictions preserved
- ✅ No regeneration = no look-ahead bias
- ✅ Reliable for backtesting

---

## Verification

To verify prediction integrity:

```bash
# Check if prediction exists in archive (original)
cat results/prediction_archive/predictions_2025-11-22.csv

# Compare with historical file (should match)
grep "2025-11-22" results/predictions_2025_26.csv
```

If they don't match → look-ahead bias detected

---

## Best Practices Going Forward

1. **Never delete archived predictions** - they're the source of truth
2. **Never regenerate historical predictions** - only append new ones
3. **Always use pre-game predictions** - from archive, not recalculated
4. **Verify data integrity** - compare archives with historical file periodically

---

## Technical Details

### Why Regeneration Causes Bias:

**Rolling Average Example:**
- **Before Nov 22 game:** Team's 5-game avg uses games from Nov 17-21
- **After Nov 22 game:** Team's 5-game avg uses games from Nov 18-22
- **If recalculated Nov 23:** Nov 22 prediction uses the post-game average (includes Nov 22 result)

This creates **information leakage** where future data influences past predictions.

### Correct Approach:

```python
# WRONG: Regenerate all predictions
all_games = historical + current_season  # Includes all games
features = build_features(all_games)  # Uses data from the future

# RIGHT: Only use data available before each game
for game in new_games:
    historical_up_to_game = all_games[all_games.date < game.date]
    features = build_features(historical_up_to_game)
    prediction = model.predict(features)
```

---

## Acknowledgment

This issue was discovered by careful user observation comparing pre-game predictions with historical records. Thank you for identifying this critical data integrity problem.

---

**Last Updated:** November 23, 2025
**Status:** Fixed via incremental-only pipeline
