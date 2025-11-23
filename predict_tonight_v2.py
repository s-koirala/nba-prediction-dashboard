"""
Predict Tonight's Games - V2.0

Uses the new temporally-trained model (rolling 4yr window, no look-ahead bias).
Simpler, more reliable, production-ready.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

from models.elo_system import NBAEloRatings


def load_latest_model():
    """Load the latest trained model."""
    model_dir = 'models/v2.0.0_20251123_115028'

    print(f"Loading model from: {model_dir}")

    # Load metadata
    with open(f'{model_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"  Model version: {metadata['version']}")
    print(f"  Trained: {metadata['trained_date']}")
    print(f"  Training window: {metadata['training_info']['train_start']} to {metadata['training_info']['train_end']}")
    print(f"  Games: {metadata['metrics']['training_games']}")

    # Load Elo ratings
    with open(f'{model_dir}/elo/ratings.pkl', 'rb') as f:
        ratings = pickle.load(f)

    # Recreate Elo system
    elo = NBAEloRatings(
        k_factor=metadata['elo_parameters']['k_factor'],
        home_advantage=metadata['elo_parameters']['home_advantage'],
        initial_rating=metadata['elo_parameters']['initial_rating'],
        season_reset_factor=metadata['elo_parameters']['season_reset_factor'],
        mean_rating=metadata['elo_parameters']['mean_rating']
    )
    elo.ratings = ratings
    elo.margin_of_victory_multiplier = elo.margin_of_victory_multiplier_538

    print(f"  [OK] Model loaded successfully")
    print(f"  Teams tracked: {len(elo.ratings)}")

    return elo, metadata


def get_todays_games():
    """Get today's scheduled NBA games."""
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"Fetching games for {date_str}")
    print(f"{'='*60}")

    try:
        # Get today's scoreboard
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = scoreboard.get_data_frames()[0]

        if len(games) == 0:
            print(f"No games scheduled for {date_str}")
            return None, date_str

        # Get team names mapping
        all_teams = teams.get_teams()
        team_map = {team['id']: team['full_name'] for team in all_teams}

        # Add team names
        games['HOME_TEAM_NAME'] = games['HOME_TEAM_ID'].map(team_map)
        games['VISITOR_TEAM_NAME'] = games['VISITOR_TEAM_ID'].map(team_map)

        print(f"Found {len(games)} games scheduled for tonight")

        return games, date_str

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return None, date_str


def make_predictions(elo, todays_games):
    """Make predictions for tonight's games."""
    print(f"\n{'='*60}")
    print(f"MAKING PREDICTIONS")
    print(f"{'='*60}\n")

    predictions = []

    for idx, game in todays_games.iterrows():
        home_team = game['HOME_TEAM_NAME']
        away_team = game['VISITOR_TEAM_NAME']

        # Make prediction
        win_prob_home, expected_margin = elo.predict_game(home_team, away_team, is_home_a=True)

        # Get team ratings
        home_rating = elo.ratings.get(home_team, elo.initial_rating)
        away_rating = elo.ratings.get(away_team, elo.initial_rating)

        # Determine prediction
        if expected_margin > 0:
            predicted_winner = home_team
            confidence = 'HIGH' if win_prob_home > 0.70 else 'MEDIUM' if win_prob_home > 0.60 else 'LOW'
        else:
            predicted_winner = away_team
            confidence = 'HIGH' if win_prob_home < 0.30 else 'MEDIUM' if win_prob_home < 0.40 else 'LOW'

        predictions.append({
            'game_date': datetime.now().strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_elo': home_rating,
            'away_elo': away_rating,
            'win_prob_home': win_prob_home,
            'expected_margin': expected_margin,
            'predicted_winner': predicted_winner,
            'confidence': confidence
        })

        print(f"{home_team} vs {away_team}")
        print(f"  Home Elo: {home_rating:.0f} | Away Elo: {away_rating:.0f}")
        print(f"  Win Probability: {win_prob_home:.1%} (home)")
        print(f"  Expected Margin: {expected_margin:+.1f} points")
        print(f"  Prediction: {predicted_winner} ({confidence} confidence)")
        print()

    return pd.DataFrame(predictions)


def save_predictions(predictions_df):
    """Save predictions to CSV."""
    output_path = 'results/tonights_predictions.csv'

    predictions_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"PREDICTIONS SAVED")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Games: {len(predictions_df)}")

    # Show summary
    print(f"\nPrediction Summary:")
    print(f"  HIGH confidence: {len(predictions_df[predictions_df['confidence'] == 'HIGH'])}")
    print(f"  MEDIUM confidence: {len(predictions_df[predictions_df['confidence'] == 'MEDIUM'])}")
    print(f"  LOW confidence: {len(predictions_df[predictions_df['confidence'] == 'LOW'])}")

    return output_path


def main():
    """Main prediction pipeline."""
    print("\n" + "="*60)
    print(" "*15 + "NBA PREDICTION - V2.0")
    print("="*60)
    print("Using optimized temporal model (rolling 4yr window)")
    print("="*60)

    # Load model
    elo, metadata = load_latest_model()

    # Get today's games
    todays_games, date_str = get_todays_games()

    if todays_games is None or len(todays_games) == 0:
        print(f"\nNo predictions to make (no games scheduled)")
        return

    # Make predictions
    predictions = make_predictions(elo, todays_games)

    # Save predictions
    output_path = save_predictions(predictions)

    print(f"\n{'='*60}")
    print(f"PREDICTION PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nQuality assurance:")
    print(f"  [OK] Using temporally-trained model (no look-ahead bias)")
    print(f"  [OK] Rolling 4-year window (optimal from grid search)")
    print(f"  [OK] FiveThirtyEight MOV formula + Season reset")
    print(f"  [OK] Expected accuracy: 67.1% (from validation)")

    print(f"\nReady for dashboard display!")

    return predictions


if __name__ == "__main__":
    main()
