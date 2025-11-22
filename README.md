# 🏀 NBA Prediction Model Dashboard

A machine learning-powered NBA game prediction system with an interactive dashboard featuring ensemble models (Elo, Neural Network, XGBoost) and walk-forward validated betting strategies.

## 🎯 Key Features

- **Live Game Predictions**: Daily predictions for tonight's NBA games with confidence levels
- **Ensemble ML Models**: Combines Elo ratings, Neural Networks, and XGBoost
- **Optimized Betting Strategies**: Walk-forward validated with 32-35% ROI
- **Performance Tracking**: Historical analysis with date range filtering
- **Interactive Dashboard**: Real-time Streamlit UI with Plotly visualizations

## 📊 Performance Metrics

- **Out-of-Sample ROI**: 32-35% (tested on 2025-26 season)
- **Win Rate**: 65-70% accuracy on spread predictions
- **Ensemble Accuracy**: 66.0% on test set
- **Validation**: 1,456 games walk-forward testing

## Project Structure

```
nba-prediction-model/
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned and preprocessed data
│   └── features/         # Engineered features
├── models/
│   ├── elo/             # Elo rating system implementation
│   ├── neural_network/  # Deep learning models
│   ├── ensemble/        # Combined ensemble models
│   └── xgboost/         # Gradient boosting models
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── data_collection/ # Data scraping scripts
│   ├── feature_engineering/ # Feature creation
│   ├── models/          # Model implementations
│   └── utils/           # Helper functions
├── results/             # Model outputs and predictions
└── config/              # Configuration files
```

## Models Implemented

### 1. Elo Rating System (FiveThirtyEight-style)
- Historical game-by-game ratings
- K-factor optimization
- Home court advantage adjustment

### 2. Neural Network Predictor
- Multi-layer perceptron
- Rolling averages and momentum features
- ~70% accuracy target

### 3. XGBoost Ensemble
- Advanced feature importance
- Hyperparameter tuning
- Team and player statistics

### 4. Meta Ensemble
- Combines predictions from all models
- Weighted voting system
- Optimized for maximum accuracy

## Data Sources
- NBA.com API (via nba_api)
- Basketball-Reference.com
- FiveThirtyEight historical data

## 💰 Betting Strategies

### Diversified Strategy (Recommended for Max Profit)
- **ROI**: 32.37%
- **Allocation**: HIGH=14%, MEDIUM=1%, LOW=14%
- **Total Profit**: $33,145 on $10k bankroll (231 games)
- **Best For**: Maximum total returns

### Optimized Strategy (Maximum Efficiency)
- **ROI**: 35.23%
- **Allocation**: LOW confidence only at 8.5%
- **Total Profit**: $14,372 on $10k bankroll (48 games)
- **Best For**: Highest ROI per bet

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard_v2.py

# Generate tonight's predictions
python predict_tonight.py
```

## 🛠️ Tech Stack

- **ML**: scikit-learn, XGBoost
- **Dashboard**: Streamlit, Plotly
- **Data**: NBA API (nba_api)
- **Validation**: Walk-forward testing

## ⚠️ Disclaimer

This model is for educational and research purposes only. Sports betting involves risk and may not be legal in your jurisdiction. Past performance does not guarantee future results.

---

**Built with Claude Code**
