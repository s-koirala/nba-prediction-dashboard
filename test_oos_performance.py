"""
Out-of-Sample Testing - 2024-25 Season
Tests trained models on completely unseen data
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from data_collection.nba_scraper import NBADataScraper
from feature_engineering.feature_builder import NBAFeatureBuilder
from models.elo_system import NBAEloRatings
from models.neural_network import NBANeuralNetwork
from models.xgboost_model import NBAXGBoost
from models.ensemble import NBAEnsemble
from utils.helpers import evaluate_spread_predictions, calculate_betting_roi, print_metrics

def scrape_oos_season(season='2024-25'):
    """
    Scrape out-of-sample season data
    """
    print("\n" + "="*60)
    print(f"SCRAPING OUT-OF-SAMPLE DATA: {season}")
    print("="*60)

    scraper = NBADataScraper(output_dir='data/oos')
    games = scraper.get_season_games(season=season)

    if games is None or len(games) == 0:
        print(f"No games found for {season} season")
        return None

    # Process games
    processed = scraper.process_game_data(games)

    # Save
    os.makedirs('data/oos', exist_ok=True)
    processed.to_csv('data/oos/games_2024_25.csv', index=False)

    print(f"\nScraped {len(processed)} games from {season} season")
    return processed

def generate_oos_features(oos_games, historical_data):
    """
    Generate features for OOS games using historical data + OOS games
    """
    print("\n" + "="*60)
    print("GENERATING OOS FEATURES")
    print("="*60)

    # Combine historical and OOS data to calculate rolling features
    all_games = pd.concat([historical_data, oos_games], ignore_index=True)
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    all_games = all_games.sort_values('GAME_DATE')

    # Build features
    feature_builder = NBAFeatureBuilder(rolling_windows=[5, 10, 20])
    features = feature_builder.build_features(all_games)

    # Calculate Elo ratings (need to start from historical data)
    elo_system = NBAEloRatings(k_factor=20, home_advantage=100)

    # First, load final Elo ratings from training data
    historical_elo = pd.read_csv('data/features/elo_ratings.csv')
    historical_elo['GAME_DATE'] = pd.to_datetime(historical_elo['GAME_DATE'])

    # Get final ratings for each team
    latest_ratings = historical_elo.sort_values('GAME_DATE').groupby('HOME_TEAM').last()
    for team in latest_ratings.index:
        elo_system.ratings[team] = latest_ratings.loc[team, 'POST_ELO_HOME']

    # Now process OOS games
    oos_elo = elo_system.process_season(oos_games)

    # Filter features to only OOS games
    oos_game_dates = set(oos_games['GAME_DATE'].dt.date)

    features['rolling']['GAME_DATE'] = pd.to_datetime(features['rolling']['GAME_DATE'])
    features['momentum']['GAME_DATE'] = pd.to_datetime(features['momentum']['GAME_DATE'])
    features['rest']['GAME_DATE'] = pd.to_datetime(features['rest']['GAME_DATE'])

    rolling_oos = features['rolling'][features['rolling']['GAME_DATE'].dt.date.isin(oos_game_dates)]
    momentum_oos = features['momentum'][features['momentum']['GAME_DATE'].dt.date.isin(oos_game_dates)]
    rest_oos = features['rest'][features['rest']['GAME_DATE'].dt.date.isin(oos_game_dates)]

    print(f"\nOOS Features generated:")
    print(f"  - Rolling stats: {len(rolling_oos)} records")
    print(f"  - Momentum: {len(momentum_oos)} records")
    print(f"  - Rest: {len(rest_oos)} records")
    print(f"  - Elo: {len(oos_elo)} records")

    return oos_games, oos_elo, rolling_oos, momentum_oos, rest_oos

def prepare_oos_features(games, elo, rolling, momentum, rest):
    """
    Prepare OOS feature matrix (same as training)
    """
    print("\nPreparing OOS feature matrix...")

    # Convert dates
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    elo['GAME_DATE'] = pd.to_datetime(elo['GAME_DATE'])
    rolling['GAME_DATE'] = pd.to_datetime(rolling['GAME_DATE'])
    momentum['GAME_DATE'] = pd.to_datetime(momentum['GAME_DATE'])
    rest['GAME_DATE'] = pd.to_datetime(rest['GAME_DATE'])

    # Merge Elo
    games = games.merge(
        elo[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'PRE_ELO_HOME', 'PRE_ELO_AWAY',
             'WIN_PROB_HOME', 'EXPECTED_MARGIN']],
        left_on=['GAME_DATE', 'TEAM_NAME_HOME', 'TEAM_NAME_AWAY'],
        right_on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
        how='left'
    )

    # Merge rolling stats HOME
    rolling_home = rolling[['GAME_DATE', 'TEAM_NAME',
                            'PTS_ROLL_5', 'FG_PCT_ROLL_5', 'FG3_PCT_ROLL_5',
                            'REB_ROLL_5', 'AST_ROLL_5', 'TOV_ROLL_5',
                            'PTS_ROLL_10', 'FG_PCT_ROLL_10', 'FG3_PCT_ROLL_10']].copy()
    rolling_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME'] + [f'{c}_HOME' for c in rolling_home.columns[2:]]
    games = games.merge(rolling_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # Merge rolling stats AWAY
    rolling_away = rolling[['GAME_DATE', 'TEAM_NAME',
                            'PTS_ROLL_5', 'FG_PCT_ROLL_5', 'FG3_PCT_ROLL_5',
                            'REB_ROLL_5', 'AST_ROLL_5', 'TOV_ROLL_5',
                            'PTS_ROLL_10', 'FG_PCT_ROLL_10', 'FG3_PCT_ROLL_10']].copy()
    rolling_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY'] + [f'{c}_AWAY' for c in rolling_away.columns[2:]]
    games = games.merge(rolling_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Merge momentum HOME
    momentum_home = momentum[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']].copy()
    momentum_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME', 'WIN_PCT_L5_HOME', 'WIN_PCT_L10_HOME', 'STREAK_HOME']
    games = games.merge(momentum_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # Merge momentum AWAY
    momentum_away = momentum[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']].copy()
    momentum_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY', 'WIN_PCT_L5_AWAY', 'WIN_PCT_L10_AWAY', 'STREAK_AWAY']
    games = games.merge(momentum_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Merge rest HOME
    rest_home = rest[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']].copy()
    rest_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME', 'REST_DAYS_HOME', 'B2B_HOME']
    games = games.merge(rest_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # Merge rest AWAY
    rest_away = rest[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']].copy()
    rest_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY', 'REST_DAYS_AWAY', 'B2B_AWAY']
    games = games.merge(rest_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Feature columns (same as training)
    feature_cols = [
        'PRE_ELO_HOME', 'PRE_ELO_AWAY',
        'PTS_ROLL_5_HOME', 'PTS_ROLL_5_AWAY',
        'FG_PCT_ROLL_5_HOME', 'FG_PCT_ROLL_5_AWAY',
        'FG3_PCT_ROLL_5_HOME', 'FG3_PCT_ROLL_5_AWAY',
        'REB_ROLL_5_HOME', 'REB_ROLL_5_AWAY',
        'AST_ROLL_5_HOME', 'AST_ROLL_5_AWAY',
        'TOV_ROLL_5_HOME', 'TOV_ROLL_5_AWAY',
        'PTS_ROLL_10_HOME', 'PTS_ROLL_10_AWAY',
        'FG_PCT_ROLL_10_HOME', 'FG_PCT_ROLL_10_AWAY',
        'FG3_PCT_ROLL_10_HOME', 'FG3_PCT_ROLL_10_AWAY',
        'WIN_PCT_L5_HOME', 'WIN_PCT_L5_AWAY',
        'WIN_PCT_L10_HOME', 'WIN_PCT_L10_AWAY',
        'STREAK_HOME', 'STREAK_AWAY',
        'REST_DAYS_HOME', 'REST_DAYS_AWAY',
        'B2B_HOME', 'B2B_AWAY'
    ]

    # Derived features
    games['ELO_DIFF'] = games['PRE_ELO_HOME'] - games['PRE_ELO_AWAY']
    games['PTS_DIFF_5'] = games['PTS_ROLL_5_HOME'] - games['PTS_ROLL_5_AWAY']
    games['FG_PCT_DIFF_5'] = games['FG_PCT_ROLL_5_HOME'] - games['FG_PCT_ROLL_5_AWAY']
    games['REST_DIFF'] = games['REST_DAYS_HOME'] - games['REST_DAYS_AWAY']
    feature_cols.extend(['ELO_DIFF', 'PTS_DIFF_5', 'FG_PCT_DIFF_5', 'REST_DIFF'])

    # Clean
    games_clean = games.dropna(subset=feature_cols + ['POINT_DIFFERENTIAL'])

    X = games_clean[feature_cols].values
    y = games_clean['POINT_DIFFERENTIAL'].values

    print(f"OOS games: {len(games_clean)}")
    print(f"Features: {X.shape}")

    return X, y, games_clean

def load_trained_models():
    """
    Load previously trained models
    """
    print("\n" + "="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)

    # Load Neural Network
    nn_model = NBANeuralNetwork()
    nn_model.load_model('models/neural_network_fixed')
    print("Neural Network loaded")

    # Load XGBoost
    xgb_model = NBAXGBoost()
    xgb_model.load_model('models/xgboost_fixed')
    print("XGBoost loaded")

    # Load Ensemble
    ensemble = NBAEnsemble()
    ensemble.load_model('models/ensemble_fixed')
    print("Ensemble loaded")

    return nn_model, xgb_model, ensemble

def evaluate_oos_performance(X_oos, y_oos, elo_oos_pred, nn_model, xgb_model, ensemble, games_clean=None):
    """
    Evaluate models on OOS data
    """
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE PERFORMANCE EVALUATION")
    print("="*60)

    results = {}

    # Elo
    print("\n[1/4] Elo Model (OOS)")
    elo_metrics = evaluate_spread_predictions(y_oos, elo_oos_pred)
    print_metrics(elo_metrics, "Elo OOS Performance")
    results['elo'] = elo_metrics

    # Neural Network
    print("\n[2/4] Neural Network (OOS)")
    nn_pred = nn_model.predict(X_oos)
    nn_metrics = evaluate_spread_predictions(y_oos, nn_pred)
    print_metrics(nn_metrics, "Neural Network OOS Performance")
    results['neural_network'] = nn_metrics

    # XGBoost
    print("\n[3/4] XGBoost (OOS)")
    xgb_pred = xgb_model.predict(X_oos)
    xgb_metrics = evaluate_spread_predictions(y_oos, xgb_pred)
    print_metrics(xgb_metrics, "XGBoost OOS Performance")
    results['xgboost'] = xgb_metrics

    # Ensemble
    print("\n[4/4] Ensemble (OOS)")
    ensemble_pred = ensemble.predict(elo_oos_pred, nn_pred, xgb_pred)
    ensemble_metrics = evaluate_spread_predictions(y_oos, ensemble_pred)
    print_metrics(ensemble_metrics, "Ensemble OOS Performance")
    results['ensemble'] = ensemble_metrics

    # Betting simulation
    print("\n" + "="*60)
    print("OOS BETTING PERFORMANCE")
    print("="*60)

    predictions_df = pd.DataFrame({
        'actual_margin': y_oos,
        'elo_pred': elo_oos_pred,
        'nn_pred': nn_pred,
        'xgb_pred': xgb_pred,
        'ensemble_pred': ensemble_pred
    })

    # Add team names if games_clean is provided
    if games_clean is not None:
        predictions_df['game_date'] = games_clean['GAME_DATE'].values
        predictions_df['home_team'] = games_clean['TEAM_NAME_HOME'].values
        predictions_df['away_team'] = games_clean['TEAM_NAME_AWAY'].values
        # Reorder columns
        predictions_df = predictions_df[['game_date', 'home_team', 'away_team', 'actual_margin',
                                         'elo_pred', 'nn_pred', 'xgb_pred', 'ensemble_pred']]

    for model_name in ['elo', 'nn', 'xgb', 'ensemble']:
        pred_col = f'{model_name}_pred'
        predictions_df['actual_cover'] = (predictions_df['actual_margin'] > 0).astype(int)
        predictions_df['predicted_cover'] = (predictions_df[pred_col] > 0).astype(int)

        betting_metrics = calculate_betting_roi(predictions_df)
        print(f"\n{model_name.upper()} OOS Betting:")
        print_metrics(betting_metrics, f"{model_name.upper()} OOS Betting")

        if betting_metrics['win_rate'] > 0.524:
            print(f"  PROFITABLE on OOS data! Win rate: {betting_metrics['win_rate']:.2%}")
        else:
            print(f"  Not profitable on OOS data. Win rate: {betting_metrics['win_rate']:.2%}")

    return results, predictions_df

def main():
    print("\n" + "="*60)
    print(" OUT-OF-SAMPLE TESTING - 2024-25 SEASON")
    print("="*60)

    # Step 1: Scrape OOS season
    oos_games = scrape_oos_season(season='2024-25')

    if oos_games is None or len(oos_games) < 10:
        print("\nNot enough OOS games available yet (season in progress)")
        print("Falling back to using last season (2023-24) as OOS test")

        # Load historical data
        historical = pd.read_csv('data/raw/games_processed.csv')
        historical['GAME_DATE'] = pd.to_datetime(historical['GAME_DATE'])

        # Use 2023-24 as OOS
        oos_games = historical[historical['GAME_DATE'].dt.year == 2024]
        historical = historical[historical['GAME_DATE'].dt.year < 2024]

    else:
        # Load historical training data
        historical = pd.read_csv('data/raw/games_processed.csv')

    # Ensure column consistency
    if 'GAME_DATE_HOME' in oos_games.columns:
        oos_games['GAME_DATE'] = oos_games['GAME_DATE_HOME']
    if 'GAME_DATE_HOME' in historical.columns:
        historical['GAME_DATE'] = historical['GAME_DATE_HOME']

    # Step 2: Generate features for OOS data
    oos_games, oos_elo, rolling_oos, momentum_oos, rest_oos = generate_oos_features(
        oos_games, historical
    )

    # Step 3: Prepare feature matrix
    X_oos, y_oos, games_clean = prepare_oos_features(
        oos_games, oos_elo, rolling_oos, momentum_oos, rest_oos
    )

    # Get Elo predictions
    elo_oos_pred = games_clean['EXPECTED_MARGIN'].values

    # Step 4: Load trained models
    nn_model, xgb_model, ensemble = load_trained_models()

    # Step 5: Evaluate on OOS data
    results, predictions_df = evaluate_oos_performance(
        X_oos, y_oos, elo_oos_pred, nn_model, xgb_model, ensemble, games_clean
    )

    # Save results
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv('results/oos_predictions.csv', index=False)
    pd.DataFrame(results).T.to_csv('results/oos_performance.csv')

    print("\n" + "="*60)
    print("OOS TESTING COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - results/oos_predictions.csv")
    print("  - results/oos_performance.csv")

    # Summary
    print("\n" + "="*60)
    print("OOS PERFORMANCE SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ATS Accuracy: {metrics['ats_accuracy']:.2%}")
        print(f"  MAE: {metrics['mae']:.2f} points")

if __name__ == "__main__":
    main()
