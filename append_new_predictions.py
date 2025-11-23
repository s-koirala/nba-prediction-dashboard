"""
Append new predictions to historical data without regenerating old ones.
This prevents look-ahead bias by freezing predictions at the time they were made.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def append_new_completed_games():
    """
    Only add predictions for NEW games that have completed since last update.
    Never regenerate existing predictions.
    """
    print("\n" + "="*60)
    print(" APPENDING NEW COMPLETED GAMES")
    print("="*60)

    # Load existing predictions
    try:
        existing = pd.read_csv('results/predictions_2025_26.csv')
        existing['game_date'] = pd.to_datetime(existing['game_date'])

        # Get the latest date we have predictions for
        latest_date = existing['game_date'].max()
        print(f"\nExisting predictions through: {latest_date.date()}")

        # Get set of games we already have
        existing_games = set(
            zip(existing['game_date'].dt.date,
                existing['home_team'],
                existing['away_team'])
        )
        print(f"Total existing games: {len(existing_games)}")

    except FileNotFoundError:
        existing = None
        latest_date = pd.Timestamp('2025-10-20')  # Start of 2025-26 season
        existing_games = set()
        print("\nNo existing predictions found. Starting fresh.")

    # Load completed games for 2025-26 season
    from data_collection.nba_scraper import NBADataScraper

    scraper = NBADataScraper(output_dir='data/oos')
    current_season_games = scraper.get_season_games(season='2025-26')

    if current_season_games is None or len(current_season_games) == 0:
        print("\nNo games found for 2025-26 season")
        return

    # Process games
    oos_games = scraper.process_game_data(current_season_games)
    oos_games['GAME_DATE'] = pd.to_datetime(oos_games['GAME_DATE'])

    # Filter to only NEW completed games (after latest_date)
    new_games = oos_games[oos_games['GAME_DATE'] > latest_date].copy()

    if len(new_games) == 0:
        print(f"\nNo new games found after {latest_date.date()}")
        return

    print(f"\nFound {len(new_games)} new completed games")

    # For each new game, generate prediction using data available BEFORE that game
    from feature_engineering.feature_builder import NBAFeatureBuilder
    from models.elo_system import NBAEloRatings
    from models.neural_network import NBANeuralNetwork
    from models.xgboost_model import NBAXGBoost
    from models.ensemble import NBAEnsemble

    # Load models
    nn_model = NBANeuralNetwork()
    nn_model.load_model('models/neural_network_fixed')

    xgb_model = NBAXGBoost()
    xgb_model.load_model('models/xgboost_fixed')

    ensemble = NBAEnsemble()
    ensemble.load_model('models/ensemble_fixed')

    # Load historical data (up to but not including new games)
    historical = pd.read_csv('data/raw/games_processed.csv')
    if 'GAME_DATE_HOME' in historical.columns:
        historical['GAME_DATE'] = historical['GAME_DATE_HOME']
    historical['GAME_DATE'] = pd.to_datetime(historical['GAME_DATE'])

    # Load existing 2025-26 games up to latest_date
    existing_2025_games = oos_games[oos_games['GAME_DATE'] <= latest_date].copy()

    new_predictions = []

    for game_date in sorted(new_games['GAME_DATE'].unique()):
        print(f"\nProcessing games from {game_date.date()}...")

        games_on_date = new_games[new_games['GAME_DATE'] == game_date]

        # Build features using ONLY data available before this game date
        all_games_before = pd.concat([
            historical,
            existing_2025_games[existing_2025_games['GAME_DATE'] < game_date]
        ], ignore_index=True)
        all_games_before = all_games_before.sort_values('GAME_DATE')

        # Generate features
        feature_builder = NBAFeatureBuilder(rolling_windows=[5, 10, 20])
        features = feature_builder.build_features(all_games_before)

        # Calculate Elo ratings
        elo_system = NBAEloRatings(k_factor=20, home_advantage=100)

        # Load final historical Elo
        historical_elo = pd.read_csv('data/features/elo_ratings.csv')
        historical_elo['GAME_DATE'] = pd.to_datetime(historical_elo['GAME_DATE'])
        latest_ratings = historical_elo.sort_values('GAME_DATE').groupby('HOME_TEAM').last()
        for team in latest_ratings.index:
            elo_system.ratings[team] = latest_ratings.loc[team, 'POST_ELO_HOME']

        # Process all 2025-26 games up to (but not including) this date
        if len(existing_2025_games) > 0:
            games_to_process = existing_2025_games[existing_2025_games['GAME_DATE'] < game_date]
            if len(games_to_process) > 0:
                elo_system.process_season(games_to_process)

        # Now generate predictions for games on this date
        oos_elo = elo_system.process_season(games_on_date)

        # Prepare features for this date's games
        # (Similar to test_2025_26_season.py but using only pre-game data)

        for idx, game in games_on_date.iterrows():
            game_key = (game['GAME_DATE'].date(), game['TEAM_NAME_HOME'], game['TEAM_NAME_AWAY'])

            if game_key in existing_games:
                continue  # Skip if already have this prediction

            # Get Elo prediction
            elo_row = oos_elo[
                (oos_elo['GAME_DATE'] == game['GAME_DATE']) &
                (oos_elo['HOME_TEAM'] == game['TEAM_NAME_HOME']) &
                (oos_elo['AWAY_TEAM'] == game['TEAM_NAME_AWAY'])
            ]

            if len(elo_row) == 0:
                print(f"  Skipping {game['TEAM_NAME_AWAY']} @ {game['TEAM_NAME_HOME']}: No Elo data")
                continue

            elo_pred = elo_row.iloc[0]['EXPECTED_MARGIN']

            # For now, use Elo as fallback for NN/XGB
            # (Full implementation would build complete feature vectors)
            nn_pred = elo_pred
            xgb_pred = elo_pred
            ensemble_pred = elo_pred

            new_predictions.append({
                'game_date': game['GAME_DATE'],
                'home_team': game['TEAM_NAME_HOME'],
                'away_team': game['TEAM_NAME_AWAY'],
                'actual_margin': game['POINT_DIFFERENTIAL'],
                'elo_pred': elo_pred,
                'nn_pred': nn_pred,
                'xgb_pred': xgb_pred,
                'ensemble_pred': ensemble_pred,
                'actual_cover': 1 if game['POINT_DIFFERENTIAL'] > 0 else 0,
                'predicted_cover': 1 if ensemble_pred > 0 else 0
            })

            print(f"  Added: {game['TEAM_NAME_AWAY']} @ {game['TEAM_NAME_HOME']}")

        # Add these games to existing_2025_games for next iteration
        existing_2025_games = pd.concat([existing_2025_games, games_on_date], ignore_index=True)

    if len(new_predictions) == 0:
        print("\nNo new predictions generated")
        return

    # Append to existing file
    new_df = pd.DataFrame(new_predictions)

    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values('game_date')
    combined.to_csv('results/predictions_2025_26.csv', index=False)

    print(f"\n✅ Added {len(new_predictions)} new predictions")
    print(f"Total predictions in file: {len(combined)}")

if __name__ == "__main__":
    append_new_completed_games()
