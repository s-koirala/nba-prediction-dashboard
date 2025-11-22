"""
Main Training Pipeline for NBA Prediction Models
Trains Elo, Neural Network, XGBoost, and Ensemble models
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from models.elo_system import NBAEloRatings
from models.neural_network import NBANeuralNetwork
from models.xgboost_model import NBAXGBoost
from models.ensemble import NBAEnsemble
from utils.helpers import evaluate_spread_predictions, calculate_betting_roi, print_metrics
from sklearn.model_selection import train_test_split

def load_and_merge_features():
    """
    Load all feature datasets and merge into single training dataframe
    """
    print("\n" + "="*60)
    print("LOADING AND MERGING FEATURES")
    print("="*60)

    # Load processed games
    games = pd.read_csv('data/raw/games_processed.csv')

    # Handle column naming
    if 'GAME_DATE_HOME' in games.columns and 'GAME_DATE' not in games.columns:
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE_HOME'])
    else:
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    print(f"Loaded {len(games)} games")

    # Load features
    try:
        elo = pd.read_csv('data/features/elo_ratings.csv')
        rolling = pd.read_csv('data/features/rolling_stats.csv')
        momentum = pd.read_csv('data/features/momentum.csv')
        rest = pd.read_csv('data/features/rest_days.csv')

        print(f"Loaded feature sets:")
        print(f"  - Elo ratings: {len(elo)} records")
        print(f"  - Rolling stats: {len(rolling)} records")
        print(f"  - Momentum: {len(momentum)} records")
        print(f"  - Rest days: {len(rest)} records")

    except FileNotFoundError as e:
        print(f"ERROR: Feature files not found. Run run_data_collection.py first.")
        print(f"Missing file: {e}")
        return None

    return games, elo, rolling, momentum, rest

def prepare_training_data(games, elo, rolling, momentum, rest):
    """
    Merge all features and prepare training dataset
    """
    print("\nPreparing training data...")

    # For simplicity, we'll use basic game stats + Elo
    # You can extend this to include all features

    # Convert date columns to datetime for proper merging
    elo['GAME_DATE'] = pd.to_datetime(elo['GAME_DATE'])

    # Merge Elo ratings
    games = games.merge(
        elo[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'PRE_ELO_HOME', 'PRE_ELO_AWAY',
             'WIN_PROB_HOME', 'EXPECTED_MARGIN']],
        left_on=['GAME_DATE', 'TEAM_NAME_HOME', 'TEAM_NAME_AWAY'],
        right_on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
        how='left'
    )

    # Create feature matrix
    feature_cols = [
        'PRE_ELO_HOME', 'PRE_ELO_AWAY',
        'FG_PCT_HOME', 'FG_PCT_AWAY',
        'FG3_PCT_HOME', 'FG3_PCT_AWAY',
        'FT_PCT_HOME', 'FT_PCT_AWAY',
        'REB_HOME', 'REB_AWAY',
        'AST_HOME', 'AST_AWAY',
        'STL_HOME', 'STL_AWAY',
        'BLK_HOME', 'BLK_AWAY',
        'TOV_HOME', 'TOV_AWAY'
    ]

    # Add Elo difference feature
    games['ELO_DIFF'] = games['PRE_ELO_HOME'] - games['PRE_ELO_AWAY']
    feature_cols.append('ELO_DIFF')

    # Drop rows with missing features
    games_clean = games.dropna(subset=feature_cols + ['POINT_DIFFERENTIAL'])

    X = games_clean[feature_cols].values
    y = games_clean['POINT_DIFFERENTIAL'].values  # Target: margin of victory

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")

    return X, y, games_clean, feature_cols

def train_all_models(X_train, y_train, X_val, y_val, feature_names):
    """
    Train all models
    """
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    models = {}

    # 1. Neural Network
    print("\n[1/2] Training Neural Network...")
    nn_model = NBANeuralNetwork(
        hidden_layers=[128, 64, 32],
        learning_rate=0.001,
        dropout_rate=0.3
    )
    nn_model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    nn_model.save_model()
    models['neural_network'] = nn_model

    # 2. XGBoost
    print("\n[2/2] Training XGBoost...")
    xgb_model = NBAXGBoost(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200
    )
    xgb_model.train(X_train, y_train, X_val, y_val)
    xgb_model.save_model()

    # Feature importance
    importance = xgb_model.get_feature_importance(feature_names)
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))

    models['xgboost'] = xgb_model

    return models

def evaluate_models(models, X_test, y_test, elo_test_pred):
    """
    Evaluate all models on test set
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    results = {}

    # Elo predictions
    print("\n[1/4] Elo Model")
    elo_metrics = evaluate_spread_predictions(y_test, elo_test_pred)
    print_metrics(elo_metrics, "Elo Model Performance")
    results['elo'] = elo_metrics

    # Neural Network
    print("\n[2/4] Neural Network")
    nn_pred = models['neural_network'].predict(X_test)
    nn_metrics = evaluate_spread_predictions(y_test, nn_pred)
    print_metrics(nn_metrics, "Neural Network Performance")
    results['neural_network'] = nn_metrics

    # XGBoost
    print("\n[3/4] XGBoost")
    xgb_pred = models['xgboost'].predict(X_test)
    xgb_metrics = evaluate_spread_predictions(y_test, xgb_pred)
    print_metrics(xgb_metrics, "XGBoost Performance")
    results['xgboost'] = xgb_metrics

    # Ensemble
    print("\n[4/4] Ensemble Model")
    ensemble = NBAEnsemble(
        elo_weight=0.25,
        nn_weight=0.35,
        xgb_weight=0.40,
        method='weighted'
    )
    ensemble_pred = ensemble.predict(elo_test_pred, nn_pred, xgb_pred)
    ensemble_metrics = evaluate_spread_predictions(y_test, ensemble_pred)
    print_metrics(ensemble_metrics, "Ensemble Model Performance")
    ensemble.save_model()
    results['ensemble'] = ensemble_metrics

    # Betting simulation
    print("\n" + "="*60)
    print("BETTING SIMULATION (Against The Spread)")
    print("="*60)

    predictions_df = pd.DataFrame({
        'actual_margin': y_test,
        'elo_pred': elo_test_pred,
        'nn_pred': nn_pred,
        'xgb_pred': xgb_pred,
        'ensemble_pred': ensemble_pred
    })

    for model_name in ['elo', 'nn', 'xgb', 'ensemble']:
        pred_col = f'{model_name}_pred'
        predictions_df['actual_cover'] = (predictions_df['actual_margin'] > 0).astype(int)
        predictions_df['predicted_cover'] = (predictions_df[pred_col] > 0).astype(int)

        betting_metrics = calculate_betting_roi(predictions_df)
        print(f"\n{model_name.upper()} Model Betting Performance:")
        print_metrics(betting_metrics, f"{model_name.upper()} Betting Metrics")

    return results, predictions_df

def main():
    print("\n" + "="*60)
    print(" NBA PREDICTION MODEL - TRAINING PIPELINE")
    print("="*60)

    # Load data
    data = load_and_merge_features()
    if data is None:
        return

    games, elo, rolling, momentum, rest = data

    # Prepare training data
    X, y, games_clean, feature_names = prepare_training_data(games, elo, rolling, momentum, rest)

    # Split by season (use last season as test)
    # For now, use simple random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} games")
    print(f"  Validation: {len(X_val)} games")
    print(f"  Test: {len(X_test)} games")

    # Train models
    models = train_all_models(X_train, y_train, X_val, y_val, feature_names)

    # For Elo, we need to get test predictions from the games dataframe
    # This is a simplified version - in production, you'd align properly
    elo_test_idx = games_clean.sample(n=len(X_test), random_state=42).index
    elo_test_pred = games_clean.loc[elo_test_idx, 'EXPECTED_MARGIN'].values

    # Evaluate
    results, predictions_df = evaluate_models(models, X_test, y_test, elo_test_pred)

    # Save results
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv('results/predictions.csv', index=False)

    results_summary = pd.DataFrame(results).T
    results_summary.to_csv('results/model_comparison.csv')

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - results/predictions.csv")
    print("  - results/model_comparison.csv")
    print("\nModels saved to models/ directory")

if __name__ == "__main__":
    main()
