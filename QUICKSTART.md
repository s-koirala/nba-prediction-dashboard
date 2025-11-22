# NBA Prediction Model - Quick Start Guide

## Overview
This project predicts NBA game outcomes with a focus on **point spread predictions** for sports betting profitability.

### Models Implemented
1. **Elo Rating System** - FiveThirtyEight-style ratings with margin of victory adjustments
2. **Neural Network** - Deep learning model for regression
3. **XGBoost** - Gradient boosting with feature importance analysis
4. **Ensemble** - Combines all three models for optimal performance

### Target Accuracy
- **Point Spread**: 55-60% (profitable at -110 odds, breakeven = 52.4%)
- **Win/Loss**: 70-85%
- **Betting ROI**: Positive expected value

---

## Installation

### Step 1: Install Dependencies
```bash
cd nba-prediction-model
pip install -r requirements.txt
```

**Required packages:**
- `nba-api` - NBA data
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `tensorflow` (optional) - Neural networks
- `pyyaml`, `tqdm` - Utilities

### Step 2: Verify Python Version
```bash
python --version  # Should be Python 3.7+
```

---

## Usage

### Full Pipeline (Recommended for First Run)

Run the complete pipeline in one go:

```bash
python run_full_pipeline.py
```

This will:
1. Scrape NBA game data (2018-2024)
2. Download FiveThirtyEight historical data
3. Build features (Elo, rolling stats, momentum, etc.)
4. Train all models
5. Evaluate performance
6. Generate results and predictions

**Estimated time**: 20-30 minutes

---

### Step-by-Step Execution

#### Step 1: Data Collection
```bash
python run_data_collection.py
```

**What it does:**
- Scrapes game data from NBA.com API
- Downloads FiveThirtyEight Elo data
- Calculates rolling averages, rest days, momentum
- Builds Elo ratings
- Saves features to `data/features/`

**Output:**
- `data/raw/games_raw.csv` - Raw game data
- `data/raw/games_processed.csv` - Processed games
- `data/features/*.csv` - Feature datasets

#### Step 2: Model Training
```bash
python run_model_training.py
```

**What it does:**
- Loads all features
- Trains Neural Network (with early stopping)
- Trains XGBoost (with cross-validation)
- Creates Ensemble model
- Evaluates on test set
- Simulates betting performance

**Output:**
- `models/neural_network/` - Saved NN model
- `models/xgboost/` - Saved XGBoost model
- `models/ensemble/` - Ensemble config
- `results/predictions.csv` - All predictions
- `results/model_comparison.csv` - Performance metrics

---

## Understanding the Results

### Model Performance Metrics

**Point Spread Metrics:**
- `ats_accuracy`: Against-the-spread win rate (target: >55%)
- `mae`: Mean absolute error in points
- `rmse`: Root mean squared error
- `correlation`: How well predictions match actual margins

**Betting Metrics:**
- `win_rate`: Percentage of correct picks
- `roi_percent`: Return on investment (positive = profit)
- `breakeven_rate`: 52.38% (need to beat this for profit)

### Sample Output
```
Ensemble Model Performance
================================================
ats_accuracy        : 0.5823  ✓ (> 52.4% breakeven)
mae                 : 8.23
rmse                : 11.45
correlation         : 0.67

Betting Simulation (100 games, $100/bet)
================================================
total_bets          : 100
correct_bets        : 58
win_rate            : 0.5800
profit              : $456.36
roi_percent         : 4.56%   ✓ Profitable!
```

---

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Seasons to scrape
data:
  start_season: "2018-19"
  end_season: "2024-25"

# Model parameters
models:
  elo:
    k_factor: 20
    home_advantage: 100

  neural_network:
    hidden_layers: [128, 64, 32]
    learning_rate: 0.001

  xgboost:
    max_depth: 6
    n_estimators: 200

  ensemble:
    elo_weight: 0.25
    nn_weight: 0.35
    xgb_weight: 0.40
```

---

## Making Predictions

### Predict Today's Games
```python
from src.models.ensemble import NBAEnsemble
from src.models.neural_network import NBANeuralNetwork
from src.models.xgboost_model import NBAXGBoost

# Load models
nn = NBANeuralNetwork()
nn.load_model('models/neural_network')

xgb = NBAXGBoost()
xgb.load_model('models/xgboost')

ensemble = NBAEnsemble()
ensemble.load_model('models/ensemble')

# Make prediction (example)
# X = [prepare your features]
# nn_pred = nn.predict(X)
# xgb_pred = xgb.predict(X)
# elo_pred = [from Elo system]

# ensemble_pred = ensemble.predict(elo_pred, nn_pred, xgb_pred)
# print(f"Predicted margin: {ensemble_pred[0]:.1f} points")
```

---

## Troubleshooting

### Issue: nba_api failing to scrape data
**Solution:** The API may have rate limits. Add delays between requests or use cached data.

### Issue: TensorFlow not installing
**Solution:** Neural network will automatically fall back to sklearn's MLPRegressor.

### Issue: Missing feature files
**Solution:** Run `python run_data_collection.py` first before training.

### Issue: Low accuracy
**Causes:**
- Not enough training data (add more seasons)
- Features not aligned properly
- Need hyperparameter tuning

**Solutions:**
- Increase training seasons in config
- Run XGBoost hyperparameter optimization
- Adjust ensemble weights

---

## Next Steps

1. **Backtest on Historical Data**: Test predictions on past seasons
2. **Add More Features**: Injuries, travel distance, playoff scenarios
3. **Compare Against Vegas Lines**: Scrape actual betting lines for comparison
4. **Live Predictions**: Automate daily predictions for upcoming games
5. **Web Dashboard**: Build visualization interface

---

## File Structure

```
nba-prediction-model/
├── data/
│   ├── raw/              # Scraped data
│   ├── processed/        # Cleaned data
│   └── features/         # Feature sets
├── models/               # Trained models
├── src/
│   ├── data_collection/  # Scrapers
│   ├── feature_engineering/
│   ├── models/           # Model implementations
│   └── utils/
├── results/              # Predictions and metrics
├── config/config.yaml    # Configuration
├── run_data_collection.py
├── run_model_training.py
└── QUICKSTART.md         # This file
```

---

## Resources

- [FiveThirtyEight NBA Elo Data](https://github.com/fivethirtyeight/data/tree/master/nba-elo)
- [nba_api Documentation](https://github.com/swar/nba_api)
- [Point Spread Betting Guide](https://www.sportsbettingdime.com/guides/strategy/point-spread-betting/)

---

**Good luck with your predictions! 🏀📈**
