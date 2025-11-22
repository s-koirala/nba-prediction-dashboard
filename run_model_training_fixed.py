"""
Fixed Training Pipeline - Uses ONLY Pre-Game Features
No data leakage!
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
    print("LOADING AND MERGING PRE-GAME FEATURES (NO LEAKAGE)")
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

def prepare_training_data_no_leakage(games, elo, rolling, momentum, rest):
    """
    Merge all PRE-GAME features and prepare training dataset
    NO IN-GAME STATISTICS - NO DATA LEAKAGE
    """
    print("\nPreparing training data (PRE-GAME FEATURES ONLY)...")

    # Convert date columns to datetime
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    elo['GAME_DATE'] = pd.to_datetime(elo['GAME_DATE'])
    rolling['GAME_DATE'] = pd.to_datetime(rolling['GAME_DATE'])
    momentum['GAME_DATE'] = pd.to_datetime(momentum['GAME_DATE'])
    rest['GAME_DATE'] = pd.to_datetime(rest['GAME_DATE'])

    # 1. Merge Elo ratings
    games = games.merge(
        elo[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'PRE_ELO_HOME', 'PRE_ELO_AWAY',
             'WIN_PROB_HOME', 'EXPECTED_MARGIN']],
        left_on=['GAME_DATE', 'TEAM_NAME_HOME', 'TEAM_NAME_AWAY'],
        right_on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
        how='left'
    )

    # 2. Merge rolling stats for HOME team
    rolling_home = rolling[['GAME_DATE', 'TEAM_NAME',
                            'PTS_ROLL_5', 'FG_PCT_ROLL_5', 'FG3_PCT_ROLL_5',
                            'REB_ROLL_5', 'AST_ROLL_5', 'TOV_ROLL_5',
                            'PTS_ROLL_10', 'FG_PCT_ROLL_10', 'FG3_PCT_ROLL_10']].copy()
    rolling_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME'] + [f'{c}_HOME' for c in rolling_home.columns[2:]]

    games = games.merge(rolling_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # 3. Merge rolling stats for AWAY team
    rolling_away = rolling[['GAME_DATE', 'TEAM_NAME',
                            'PTS_ROLL_5', 'FG_PCT_ROLL_5', 'FG3_PCT_ROLL_5',
                            'REB_ROLL_5', 'AST_ROLL_5', 'TOV_ROLL_5',
                            'PTS_ROLL_10', 'FG_PCT_ROLL_10', 'FG3_PCT_ROLL_10']].copy()
    rolling_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY'] + [f'{c}_AWAY' for c in rolling_away.columns[2:]]

    games = games.merge(rolling_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # 4. Merge momentum for HOME team
    momentum_home = momentum[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']].copy()
    momentum_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME', 'WIN_PCT_L5_HOME', 'WIN_PCT_L10_HOME', 'STREAK_HOME']

    games = games.merge(momentum_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # 5. Merge momentum for AWAY team
    momentum_away = momentum[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']].copy()
    momentum_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY', 'WIN_PCT_L5_AWAY', 'WIN_PCT_L10_AWAY', 'STREAK_AWAY']

    games = games.merge(momentum_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # 6. Merge rest days for HOME team
    rest_home = rest[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']].copy()
    rest_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME', 'REST_DAYS_HOME', 'B2B_HOME']

    games = games.merge(rest_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # 7. Merge rest days for AWAY team
    rest_away = rest[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']].copy()
    rest_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY', 'REST_DAYS_AWAY', 'B2B_AWAY']

    games = games.merge(rest_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Define PRE-GAME ONLY features
    feature_cols = [
        # Elo ratings
        'PRE_ELO_HOME', 'PRE_ELO_AWAY',

        # Rolling averages (5 games)
        'PTS_ROLL_5_HOME', 'PTS_ROLL_5_AWAY',
        'FG_PCT_ROLL_5_HOME', 'FG_PCT_ROLL_5_AWAY',
        'FG3_PCT_ROLL_5_HOME', 'FG3_PCT_ROLL_5_AWAY',
        'REB_ROLL_5_HOME', 'REB_ROLL_5_AWAY',
        'AST_ROLL_5_HOME', 'AST_ROLL_5_AWAY',
        'TOV_ROLL_5_HOME', 'TOV_ROLL_5_AWAY',

        # Rolling averages (10 games)
        'PTS_ROLL_10_HOME', 'PTS_ROLL_10_AWAY',
        'FG_PCT_ROLL_10_HOME', 'FG_PCT_ROLL_10_AWAY',
        'FG3_PCT_ROLL_10_HOME', 'FG3_PCT_ROLL_10_AWAY',

        # Momentum
        'WIN_PCT_L5_HOME', 'WIN_PCT_L5_AWAY',
        'WIN_PCT_L10_HOME', 'WIN_PCT_L10_AWAY',
        'STREAK_HOME', 'STREAK_AWAY',

        # Rest
        'REST_DAYS_HOME', 'REST_DAYS_AWAY',
        'B2B_HOME', 'B2B_AWAY'
    ]

    # Add derived features
    games['ELO_DIFF'] = games['PRE_ELO_HOME'] - games['PRE_ELO_AWAY']
    games['PTS_DIFF_5'] = games['PTS_ROLL_5_HOME'] - games['PTS_ROLL_5_AWAY']
    games['FG_PCT_DIFF_5'] = games['FG_PCT_ROLL_5_HOME'] - games['FG_PCT_ROLL_5_AWAY']
    games['REST_DIFF'] = games['REST_DAYS_HOME'] - games['REST_DAYS_AWAY']

    feature_cols.extend(['ELO_DIFF', 'PTS_DIFF_5', 'FG_PCT_DIFF_5', 'REST_DIFF'])

    # Drop rows with missing features (first few games of season)
    print(f"\nBefore cleaning: {len(games)} games")
    games_clean = games.dropna(subset=feature_cols + ['POINT_DIFFERENTIAL'])
    print(f"After cleaning: {len(games_clean)} games ({len(games) - len(games_clean)} removed due to insufficient history)")

    X = games_clean[feature_cols].values
    y = games_clean['POINT_DIFFERENTIAL'].values  # Target: margin of victory

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"\nFeature categories:")
    print(f"  - Elo: 2 features")
    print(f"  - Rolling averages (5-game): 12 features")
    print(f"  - Rolling averages (10-game): 6 features")
    print(f"  - Momentum: 6 features")
    print(f"  - Rest: 4 features")
    print(f"  - Derived: 4 features")
    print(f"  - TOTAL: {len(feature_cols)} PRE-GAME features")

    return X, y, games_clean, feature_cols

def train_all_models(X_train, y_train, X_val, y_val, feature_names):
    """
    Train all models
    """
    print("\n" + "="*60)
    print("TRAINING MODELS (NO DATA LEAKAGE)")
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
    nn_model.save_model('models/neural_network_fixed')
    models['neural_network'] = nn_model

    # 2. XGBoost
    print("\n[2/2] Training XGBoost...")
    xgb_model = NBAXGBoost(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200
    )
    xgb_model.train(X_train, y_train, X_val, y_val, verbose=False)
    xgb_model.save_model('models/xgboost_fixed')

    # Feature importance
    importance = xgb_model.get_feature_importance(feature_names)
    print("\nTop 15 Most Important Features:")
    print(importance.head(15))

    models['xgboost'] = xgb_model

    return models

def evaluate_models(models, X_test, y_test, elo_test_pred):
    """
    Evaluate all models on test set
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION (REALISTIC PERFORMANCE)")
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
    ensemble.save_model('models/ensemble_fixed')
    results['ensemble'] = ensemble_metrics

    # Betting simulation
    print("\n" + "="*60)
    print("BETTING SIMULATION (Against The Spread)")
    print("Target: >52.4% for profitability at -110 odds")
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

        # Profitability check
        if betting_metrics['win_rate'] > 0.524:
            print(f"  PROFITABLE! Win rate {betting_metrics['win_rate']:.1%} > 52.4% breakeven")
        else:
            print(f"  Not profitable. Need {52.4 - betting_metrics['win_rate']*100:.1f}% more accuracy")

    return results, predictions_df

def main():
    print("\n" + "="*60)
    print(" NBA PREDICTION MODEL - FIXED TRAINING PIPELINE")
    print(" NO DATA LEAKAGE - PRE-GAME FEATURES ONLY")
    print("="*60)

    # Load data
    data = load_and_merge_features()
    if data is None:
        return

    games, elo, rolling, momentum, rest = data

    # Prepare training data (NO LEAKAGE!)
    X, y, games_clean, feature_names = prepare_training_data_no_leakage(
        games, elo, rolling, momentum, rest
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} games")
    print(f"  Validation: {len(X_val)} games")
    print(f"  Test: {len(X_test)} games")

    # Train models
    models = train_all_models(X_train, y_train, X_val, y_val, feature_names)

    # Get Elo test predictions
    elo_test_idx = games_clean.sample(n=len(X_test), random_state=42).index
    elo_test_pred = games_clean.loc[elo_test_idx, 'EXPECTED_MARGIN'].values

    # Evaluate
    results, predictions_df = evaluate_models(models, X_test, y_test, elo_test_pred)

    # Save results
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv('results/predictions_fixed.csv', index=False)

    results_summary = pd.DataFrame(results).T
    results_summary.to_csv('results/model_comparison_fixed.csv')

    print("\n" + "="*60)
    print("TRAINING COMPLETE - NO DATA LEAKAGE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - results/predictions_fixed.csv")
    print("  - results/model_comparison_fixed.csv")
    print("\nFixed models saved to models/*_fixed/ directories")

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ATS Accuracy: {metrics['ats_accuracy']:.2%}")
        print(f"  MAE: {metrics['mae']:.2f} points")
        print(f"  RMSE: {metrics['rmse']:.2f} points")

if __name__ == "__main__":
    main()
