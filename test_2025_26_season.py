"""
Live Testing - 2025-26 Season (Current Season)
Tests on the most recent, truly out-of-sample data
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

def main():
    print("\n" + "="*60)
    print(" LIVE TESTING - 2025-26 NBA SEASON")
    print(" Testing on Current Season Games")
    print("="*60)

    # Scrape 2025-26 season
    print("\nScraping 2025-26 season data...")
    scraper = NBADataScraper(output_dir='data/oos')
    current_season_games = scraper.get_season_games(season='2025-26')

    if current_season_games is None or len(current_season_games) == 0:
        print("\nNo games found for 2025-26 season.")
        print("The season may not have started yet or there's an API issue.")
        return

    # Process games
    oos_games = scraper.process_game_data(current_season_games)
    print(f"\nFound {len(oos_games)} completed games in 2025-26 season")

    if len(oos_games) < 10:
        print(f"\nOnly {len(oos_games)} games available - too early in season for reliable testing")
        print("Need at least 10 games for meaningful evaluation")
        return

    # Save
    os.makedirs('data/oos', exist_ok=True)
    oos_games.to_csv('data/oos/games_2025_26.csv', index=False)

    # Load all historical training data (2018-2024)
    print("\nLoading historical training data...")
    historical = pd.read_csv('data/raw/games_processed.csv')

    # Ensure date columns
    if 'GAME_DATE_HOME' in oos_games.columns:
        oos_games['GAME_DATE'] = oos_games['GAME_DATE_HOME']
    if 'GAME_DATE_HOME' in historical.columns:
        historical['GAME_DATE'] = historical['GAME_DATE_HOME']

    oos_games['GAME_DATE'] = pd.to_datetime(oos_games['GAME_DATE'])
    historical['GAME_DATE'] = pd.to_datetime(historical['GAME_DATE'])

    # Generate features
    print("\n" + "="*60)
    print("GENERATING FEATURES FOR 2025-26 SEASON")
    print("="*60)

    # Combine historical + current season for rolling stats
    all_games = pd.concat([historical, oos_games], ignore_index=True)
    all_games = all_games.sort_values('GAME_DATE')

    # Build features
    feature_builder = NBAFeatureBuilder(rolling_windows=[5, 10, 20])
    features = feature_builder.build_features(all_games)

    # Calculate Elo ratings starting from historical final ratings
    elo_system = NBAEloRatings(k_factor=20, home_advantage=100)

    # Load final historical Elo ratings
    historical_elo = pd.read_csv('data/features/elo_ratings.csv')
    historical_elo['GAME_DATE'] = pd.to_datetime(historical_elo['GAME_DATE'])

    # Initialize with final ratings from training data
    latest_ratings = historical_elo.sort_values('GAME_DATE').groupby('HOME_TEAM').last()
    for team in latest_ratings.index:
        elo_system.ratings[team] = latest_ratings.loc[team, 'POST_ELO_HOME']

    # Process 2025-26 season
    oos_elo = elo_system.process_season(oos_games)

    # Filter features to 2025-26 games only
    oos_game_dates = set(oos_games['GAME_DATE'].dt.date)

    features['rolling']['GAME_DATE'] = pd.to_datetime(features['rolling']['GAME_DATE'])
    features['momentum']['GAME_DATE'] = pd.to_datetime(features['momentum']['GAME_DATE'])
    features['rest']['GAME_DATE'] = pd.to_datetime(features['rest']['GAME_DATE'])

    rolling_oos = features['rolling'][features['rolling']['GAME_DATE'].dt.date.isin(oos_game_dates)]
    momentum_oos = features['momentum'][features['momentum']['GAME_DATE'].dt.date.isin(oos_game_dates)]
    rest_oos = features['rest'][features['rest']['GAME_DATE'].dt.date.isin(oos_game_dates)]

    print(f"\nFeatures for {len(oos_games)} games generated successfully")

    # Prepare feature matrix
    print("\nPreparing feature matrix...")

    # Merge Elo
    oos_games = oos_games.merge(
        oos_elo[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'PRE_ELO_HOME', 'PRE_ELO_AWAY',
                 'WIN_PROB_HOME', 'EXPECTED_MARGIN']],
        left_on=['GAME_DATE', 'TEAM_NAME_HOME', 'TEAM_NAME_AWAY'],
        right_on=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'],
        how='left'
    )

    # Merge rolling stats HOME
    rolling_home = rolling_oos[['GAME_DATE', 'TEAM_NAME',
                                'PTS_ROLL_5', 'FG_PCT_ROLL_5', 'FG3_PCT_ROLL_5',
                                'REB_ROLL_5', 'AST_ROLL_5', 'TOV_ROLL_5',
                                'PTS_ROLL_10', 'FG_PCT_ROLL_10', 'FG3_PCT_ROLL_10']].copy()
    rolling_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME'] + [f'{c}_HOME' for c in rolling_home.columns[2:]]
    oos_games = oos_games.merge(rolling_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # Merge rolling stats AWAY
    rolling_away = rolling_oos[['GAME_DATE', 'TEAM_NAME',
                                'PTS_ROLL_5', 'FG_PCT_ROLL_5', 'FG3_PCT_ROLL_5',
                                'REB_ROLL_5', 'AST_ROLL_5', 'TOV_ROLL_5',
                                'PTS_ROLL_10', 'FG_PCT_ROLL_10', 'FG3_PCT_ROLL_10']].copy()
    rolling_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY'] + [f'{c}_AWAY' for c in rolling_away.columns[2:]]
    oos_games = oos_games.merge(rolling_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Merge momentum HOME
    momentum_home = momentum_oos[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']].copy()
    momentum_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME', 'WIN_PCT_L5_HOME', 'WIN_PCT_L10_HOME', 'STREAK_HOME']
    oos_games = oos_games.merge(momentum_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # Merge momentum AWAY
    momentum_away = momentum_oos[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']].copy()
    momentum_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY', 'WIN_PCT_L5_AWAY', 'WIN_PCT_L10_AWAY', 'STREAK_AWAY']
    oos_games = oos_games.merge(momentum_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Merge rest HOME
    rest_home = rest_oos[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']].copy()
    rest_home.columns = ['GAME_DATE', 'TEAM_NAME_HOME', 'REST_DAYS_HOME', 'B2B_HOME']
    oos_games = oos_games.merge(rest_home, on=['GAME_DATE', 'TEAM_NAME_HOME'], how='left')

    # Merge rest AWAY
    rest_away = rest_oos[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']].copy()
    rest_away.columns = ['GAME_DATE', 'TEAM_NAME_AWAY', 'REST_DAYS_AWAY', 'B2B_AWAY']
    oos_games = oos_games.merge(rest_away, on=['GAME_DATE', 'TEAM_NAME_AWAY'], how='left')

    # Define features
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
    oos_games['ELO_DIFF'] = oos_games['PRE_ELO_HOME'] - oos_games['PRE_ELO_AWAY']
    oos_games['PTS_DIFF_5'] = oos_games['PTS_ROLL_5_HOME'] - oos_games['PTS_ROLL_5_AWAY']
    oos_games['FG_PCT_DIFF_5'] = oos_games['FG_PCT_ROLL_5_HOME'] - oos_games['FG_PCT_ROLL_5_AWAY']
    oos_games['REST_DIFF'] = oos_games['REST_DAYS_HOME'] - oos_games['REST_DAYS_AWAY']
    feature_cols.extend(['ELO_DIFF', 'PTS_DIFF_5', 'FG_PCT_DIFF_5', 'REST_DIFF'])

    # Clean and prepare
    games_clean = oos_games.dropna(subset=feature_cols + ['POINT_DIFFERENTIAL'])

    X_oos = games_clean[feature_cols].values
    y_oos = games_clean['POINT_DIFFERENTIAL'].values
    elo_oos_pred = games_clean['EXPECTED_MARGIN'].values

    print(f"Clean games for testing: {len(games_clean)}")

    # Load trained models
    print("\n" + "="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)

    nn_model = NBANeuralNetwork()
    nn_model.load_model('models/neural_network_fixed')
    print("Neural Network loaded")

    xgb_model = NBAXGBoost()
    xgb_model.load_model('models/xgboost_fixed')
    print("XGBoost loaded")

    ensemble = NBAEnsemble()
    ensemble.load_model('models/ensemble_fixed')
    print("Ensemble loaded")

    # Make predictions
    print("\n" + "="*60)
    print("2025-26 SEASON PERFORMANCE")
    print("="*60)

    results = {}

    # Elo
    print("\n[1/4] Elo Model")
    elo_metrics = evaluate_spread_predictions(y_oos, elo_oos_pred)
    print_metrics(elo_metrics, "Elo 2025-26 Performance")
    results['elo'] = elo_metrics

    # Neural Network
    print("\n[2/4] Neural Network")
    nn_pred = nn_model.predict(X_oos)
    nn_metrics = evaluate_spread_predictions(y_oos, nn_pred)
    print_metrics(nn_metrics, "Neural Network 2025-26 Performance")
    results['neural_network'] = nn_metrics

    # XGBoost
    print("\n[3/4] XGBoost")
    xgb_pred = xgb_model.predict(X_oos)
    xgb_metrics = evaluate_spread_predictions(y_oos, xgb_pred)
    print_metrics(xgb_metrics, "XGBoost 2025-26 Performance")
    results['xgboost'] = xgb_metrics

    # Ensemble
    print("\n[4/4] Ensemble")
    ensemble_pred = ensemble.predict(elo_oos_pred, nn_pred, xgb_pred)
    ensemble_metrics = evaluate_spread_predictions(y_oos, ensemble_pred)
    print_metrics(ensemble_metrics, "Ensemble 2025-26 Performance")
    results['ensemble'] = ensemble_metrics

    # Betting simulation
    print("\n" + "="*60)
    print("2025-26 BETTING PERFORMANCE")
    print("="*60)

    predictions_df = pd.DataFrame({
        'game_date': games_clean['GAME_DATE'],
        'home_team': games_clean['TEAM_NAME_HOME'],
        'away_team': games_clean['TEAM_NAME_AWAY'],
        'actual_margin': y_oos,
        'elo_pred': elo_oos_pred,
        'nn_pred': nn_pred,
        'xgb_pred': xgb_pred,
        'ensemble_pred': ensemble_pred
    })

    for model_name in ['elo', 'nn', 'xgb', 'ensemble']:
        pred_col = f'{model_name}_pred'
        predictions_df['actual_cover'] = (predictions_df['actual_margin'] > 0).astype(int)
        predictions_df['predicted_cover'] = (predictions_df[pred_col] > 0).astype(int)

        betting_metrics = calculate_betting_roi(predictions_df)
        print(f"\n{model_name.upper()} 2025-26 Betting:")
        print_metrics(betting_metrics, f"{model_name.upper()} Betting")

        if betting_metrics['win_rate'] > 0.524:
            print(f"  PROFITABLE! Win rate: {betting_metrics['win_rate']:.2%} (need 52.4%)")
        else:
            print(f"  Not profitable yet. Win rate: {betting_metrics['win_rate']:.2%} (need 52.4%)")

    # Save results
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv('results/predictions_2025_26.csv', index=False)
    pd.DataFrame(results).T.to_csv('results/performance_2025_26.csv')

    print("\n" + "="*60)
    print("2025-26 SEASON TESTING COMPLETE")
    print("="*60)
    print(f"\nTested on {len(games_clean)} games from current season")
    print("\nResults saved to:")
    print("  - results/predictions_2025_26.csv")
    print("  - results/performance_2025_26.csv")

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY - 2025-26 SEASON")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ATS Accuracy: {metrics['ats_accuracy']:.2%}")
        print(f"  MAE: {metrics['mae']:.2f} points")
        print(f"  RMSE: {metrics['rmse']:.2f} points")

if __name__ == "__main__":
    main()
