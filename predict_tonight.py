"""
Get predictions for tonight's NBA games
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams
from data_collection.nba_scraper import NBADataScraper
from feature_engineering.feature_builder import NBAFeatureBuilder
from models.elo_system import NBAEloRatings
from models.neural_network import NBANeuralNetwork
from models.xgboost_model import NBAXGBoost
from models.ensemble import NBAEnsemble
import time

def get_todays_games():
    """
    Get today's scheduled NBA games
    """
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')

    print(f"\nFetching games for {date_str}...")

    try:
        # Get today's scoreboard
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = scoreboard.get_data_frames()[0]

        if len(games) == 0:
            print(f"No games scheduled for {date_str}")
            return None

        # Get team names mapping
        all_teams = teams.get_teams()
        team_map = {team['id']: team['full_name'] for team in all_teams}

        # Add team names to games
        games['HOME_TEAM_NAME'] = games['HOME_TEAM_ID'].map(team_map)
        games['VISITOR_TEAM_NAME'] = games['VISITOR_TEAM_ID'].map(team_map)

        print(f"Found {len(games)} games scheduled for tonight\n")
        return games

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return None

def prepare_todays_features(todays_games):
    """
    Prepare features for today's games using historical data
    """
    print("Preparing features for tonight's games...\n")

    # Load historical data
    historical = pd.read_csv('data/raw/games_processed.csv')
    if 'GAME_DATE_HOME' in historical.columns:
        historical['GAME_DATE'] = historical['GAME_DATE_HOME']
    historical['GAME_DATE'] = pd.to_datetime(historical['GAME_DATE'])

    # Load latest features
    rolling = pd.read_csv('data/features/rolling_stats.csv')
    momentum = pd.read_csv('data/features/momentum.csv')
    rest = pd.read_csv('data/features/rest_days.csv')

    # Load Elo ratings
    elo_df = pd.read_csv('data/features/elo_ratings.csv')
    elo_df['GAME_DATE'] = pd.to_datetime(elo_df['GAME_DATE'])

    # Get latest Elo ratings for each team
    latest_elo = elo_df.sort_values('GAME_DATE').groupby('HOME_TEAM').last()

    # Get latest rolling stats for each team
    rolling['GAME_DATE'] = pd.to_datetime(rolling['GAME_DATE'])
    latest_rolling = rolling.sort_values('GAME_DATE').groupby('TEAM_NAME').last()

    # Get latest momentum for each team
    momentum['GAME_DATE'] = pd.to_datetime(momentum['GAME_DATE'])
    latest_momentum = momentum.sort_values('GAME_DATE').groupby('TEAM').last()

    # Get latest rest for each team
    rest['GAME_DATE'] = pd.to_datetime(rest['GAME_DATE'])
    latest_rest = rest.sort_values('GAME_DATE').groupby('TEAM').last()

    # Prepare feature matrix for each game
    game_features = []

    for idx, game in todays_games.iterrows():
        home_team = game['HOME_TEAM_NAME']
        away_team = game['VISITOR_TEAM_NAME']

        try:
            # Get Elo ratings
            home_elo = latest_elo.loc[home_team, 'POST_ELO_HOME']
            away_elo = latest_elo.loc[away_team, 'POST_ELO_AWAY'] if away_team in latest_elo.index else latest_elo.loc[away_team, 'POST_ELO_HOME']

            # Calculate Elo expected margin
            elo_diff = home_elo - away_elo + 100  # Add home court advantage
            expected_margin = elo_diff / 25

            # Get rolling stats
            home_rolling = latest_rolling.loc[home_team] if home_team in latest_rolling.index else None
            away_rolling = latest_rolling.loc[away_team] if away_team in latest_rolling.index else None

            # Get momentum
            home_mom = latest_momentum.loc[home_team] if home_team in latest_momentum.index else None
            away_mom = latest_momentum.loc[away_team] if away_team in latest_momentum.index else None

            # Get rest
            home_rest = latest_rest.loc[home_team] if home_team in latest_rest.index else None
            away_rest = latest_rest.loc[away_team] if away_team in latest_rest.index else None

            # Build feature vector
            if all([home_rolling is not None, away_rolling is not None,
                   home_mom is not None, away_mom is not None,
                   home_rest is not None, away_rest is not None]):

                features = {
                    'HOME_TEAM': home_team,
                    'AWAY_TEAM': away_team,
                    'GAME_TIME': game['GAME_STATUS_TEXT'],

                    # Elo
                    'PRE_ELO_HOME': home_elo,
                    'PRE_ELO_AWAY': away_elo,
                    'ELO_EXPECTED_MARGIN': expected_margin,

                    # Rolling 5
                    'PTS_ROLL_5_HOME': home_rolling['PTS_ROLL_5'],
                    'PTS_ROLL_5_AWAY': away_rolling['PTS_ROLL_5'],
                    'FG_PCT_ROLL_5_HOME': home_rolling['FG_PCT_ROLL_5'],
                    'FG_PCT_ROLL_5_AWAY': away_rolling['FG_PCT_ROLL_5'],
                    'FG3_PCT_ROLL_5_HOME': home_rolling['FG3_PCT_ROLL_5'],
                    'FG3_PCT_ROLL_5_AWAY': away_rolling['FG3_PCT_ROLL_5'],
                    'REB_ROLL_5_HOME': home_rolling['REB_ROLL_5'],
                    'REB_ROLL_5_AWAY': away_rolling['REB_ROLL_5'],
                    'AST_ROLL_5_HOME': home_rolling['AST_ROLL_5'],
                    'AST_ROLL_5_AWAY': away_rolling['AST_ROLL_5'],
                    'TOV_ROLL_5_HOME': home_rolling['TOV_ROLL_5'],
                    'TOV_ROLL_5_AWAY': away_rolling['TOV_ROLL_5'],

                    # Rolling 10
                    'PTS_ROLL_10_HOME': home_rolling['PTS_ROLL_10'],
                    'PTS_ROLL_10_AWAY': away_rolling['PTS_ROLL_10'],
                    'FG_PCT_ROLL_10_HOME': home_rolling['FG_PCT_ROLL_10'],
                    'FG_PCT_ROLL_10_AWAY': away_rolling['FG_PCT_ROLL_10'],
                    'FG3_PCT_ROLL_10_HOME': home_rolling['FG3_PCT_ROLL_10'],
                    'FG3_PCT_ROLL_10_AWAY': away_rolling['FG3_PCT_ROLL_10'],

                    # Momentum
                    'WIN_PCT_L5_HOME': home_mom['WIN_PCT_L5'],
                    'WIN_PCT_L5_AWAY': away_mom['WIN_PCT_L5'],
                    'WIN_PCT_L10_HOME': home_mom['WIN_PCT_L10'],
                    'WIN_PCT_L10_AWAY': away_mom['WIN_PCT_L10'],
                    'STREAK_HOME': home_mom['STREAK'],
                    'STREAK_AWAY': away_mom['STREAK'],

                    # Rest
                    'REST_DAYS_HOME': home_rest['REST_DAYS'],
                    'REST_DAYS_AWAY': away_rest['REST_DAYS'],
                    'B2B_HOME': home_rest['IS_BACK_TO_BACK'],
                    'B2B_AWAY': away_rest['IS_BACK_TO_BACK'],

                    # Derived
                    'ELO_DIFF': home_elo - away_elo,
                    'PTS_DIFF_5': home_rolling['PTS_ROLL_5'] - away_rolling['PTS_ROLL_5'],
                    'FG_PCT_DIFF_5': home_rolling['FG_PCT_ROLL_5'] - away_rolling['FG_PCT_ROLL_5'],
                    'REST_DIFF': home_rest['REST_DAYS'] - away_rest['REST_DAYS']
                }

                game_features.append(features)

        except Exception as e:
            print(f"Warning: Could not prepare features for {home_team} vs {away_team}: {e}")
            continue

    if len(game_features) == 0:
        print("Could not prepare features for any games")
        return None

    return pd.DataFrame(game_features)

def predict_games(features_df):
    """
    Make predictions for tonight's games
    """
    print("Loading models and making predictions...\n")

    # Load models
    nn_model = NBANeuralNetwork()
    nn_model.load_model('models/neural_network_fixed')

    xgb_model = NBAXGBoost()
    xgb_model.load_model('models/xgboost_fixed')

    ensemble = NBAEnsemble()
    ensemble.load_model('models/ensemble_fixed')

    # Feature columns (same order as training)
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
        'B2B_HOME', 'B2B_AWAY',
        'ELO_DIFF', 'PTS_DIFF_5', 'FG_PCT_DIFF_5', 'REST_DIFF'
    ]

    X = features_df[feature_cols].values

    # Make predictions
    elo_pred = features_df['ELO_EXPECTED_MARGIN'].values
    nn_pred = nn_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    ensemble_pred = ensemble.predict(elo_pred, nn_pred, xgb_pred)

    # Add predictions to dataframe
    features_df['ELO_PREDICTION'] = elo_pred
    features_df['NN_PREDICTION'] = nn_pred
    features_df['XGB_PREDICTION'] = xgb_pred
    features_df['ENSEMBLE_PREDICTION'] = ensemble_pred

    return features_df

def display_predictions(predictions_df):
    """
    Display predictions in a nice format
    """
    print("\n" + "="*80)
    print("TONIGHT'S NBA GAME PREDICTIONS")
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    print("="*80)

    for idx, game in predictions_df.iterrows():
        print(f"\n{game['AWAY_TEAM']} @ {game['HOME_TEAM']}")
        print(f"Time: {game['GAME_TIME']}")
        print("-" * 80)

        # Model predictions
        print(f"\nPREDICTED MARGINS (Positive = Home Win):")
        print(f"  Elo Model:       {game['ELO_PREDICTION']:+.1f} points")
        print(f"  Neural Network:  {game['NN_PREDICTION']:+.1f} points")
        print(f"  XGBoost:         {game['XGB_PREDICTION']:+.1f} points")
        print(f"  ENSEMBLE:        {game['ENSEMBLE_PREDICTION']:+.1f} points")

        # Consensus pick
        avg_pred = (game['ELO_PREDICTION'] + game['NN_PREDICTION'] +
                   game['XGB_PREDICTION'] + game['ENSEMBLE_PREDICTION']) / 4

        if avg_pred > 0:
            winner = game['HOME_TEAM']
            margin = abs(avg_pred)
        else:
            winner = game['AWAY_TEAM']
            margin = abs(avg_pred)

        print(f"\nCONSENSUS PICK: {winner} by {margin:.1f} points")

        # Team stats
        print(f"\nTEAM STATS:")
        print(f"  {game['HOME_TEAM']:20s} - Elo: {game['PRE_ELO_HOME']:.0f} | L5: {game['WIN_PCT_L5_HOME']:.1%} | Streak: {game['STREAK_HOME']:+.0f}")
        print(f"  {game['AWAY_TEAM']:20s} - Elo: {game['PRE_ELO_AWAY']:.0f} | L5: {game['WIN_PCT_L5_AWAY']:.1%} | Streak: {game['STREAK_AWAY']:+.0f}")

        print("=" * 80)

    # Save to file
    predictions_df[['HOME_TEAM', 'AWAY_TEAM', 'GAME_TIME',
                    'ELO_PREDICTION', 'NN_PREDICTION', 'XGB_PREDICTION',
                    'ENSEMBLE_PREDICTION']].to_csv('results/tonights_predictions.csv', index=False)

    print(f"\nPredictions saved to results/tonights_predictions.csv")

def main():
    print("\n" + "="*80)
    print("NBA GAME PREDICTIONS FOR TONIGHT")
    print("="*80)

    # Get today's games
    todays_games = get_todays_games()

    if todays_games is None or len(todays_games) == 0:
        print("\nNo games scheduled for tonight.")
        return

    # Prepare features
    features_df = prepare_todays_features(todays_games)

    if features_df is None:
        print("Could not prepare features for tonight's games")
        return

    # Make predictions
    predictions_df = predict_games(features_df)

    # Display results
    display_predictions(predictions_df)

    print("\n" + "="*80)
    print(f"PREDICTION COMPLETE - {len(predictions_df)} games analyzed")
    print("="*80)

if __name__ == "__main__":
    main()
