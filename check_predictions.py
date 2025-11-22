"""
Check tonight's predictions against actual results
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

def get_live_scores():
    """
    Get live/final scores for today's games
    """
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\nFetching live scores for {today}...")

    try:
        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        games = scoreboard.get_data_frames()[0]
        line_score = scoreboard.get_data_frames()[1]

        if len(games) == 0:
            print("No games found for today")
            return None

        # Map team IDs to names
        all_teams = teams.get_teams()
        team_map = {team['id']: team['full_name'] for team in all_teams}

        results = []
        for idx, game in games.iterrows():
            home_team = team_map[game['HOME_TEAM_ID']]
            away_team = team_map[game['VISITOR_TEAM_ID']]
            status = game['GAME_STATUS_TEXT']

            # Get scores from line_score dataframe
            game_line = line_score[line_score['GAME_ID'] == game['GAME_ID']]

            home_score = None
            away_score = None

            if len(game_line) > 0:
                home_line = game_line[game_line['TEAM_ID'] == game['HOME_TEAM_ID']]
                away_line = game_line[game_line['TEAM_ID'] == game['VISITOR_TEAM_ID']]

                if len(home_line) > 0:
                    home_score = home_line.iloc[0]['PTS']
                if len(away_line) > 0:
                    away_score = away_line.iloc[0]['PTS']

            results.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'status': status,
                'game_id': game['GAME_ID']
            })

        return pd.DataFrame(results)

    except Exception as e:
        print(f"Error fetching live scores: {e}")
        return None

def compare_predictions():
    """
    Compare predictions to actual results
    """
    print("\n" + "="*80)
    print("CHECKING TONIGHT'S PREDICTIONS")
    print("="*80)

    # Load predictions
    try:
        predictions = pd.read_csv('results/tonights_predictions.csv')
    except:
        print("\nNo predictions file found. Run predict_tonight.py first.")
        return

    # Get live scores
    live_scores = get_live_scores()

    if live_scores is None:
        return

    # Merge predictions with live scores
    results = predictions.merge(
        live_scores,
        left_on=['HOME_TEAM', 'AWAY_TEAM'],
        right_on=['home_team', 'away_team'],
        how='left'
    )

    print(f"\nFound {len(results)} games\n")

    completed = 0
    in_progress = 0
    not_started = 0
    correct_picks = 0
    total_picks = 0

    for idx, game in results.iterrows():
        print("="*80)
        print(f"{game['AWAY_TEAM']} @ {game['HOME_TEAM']}")
        print(f"Status: {game['status']}")
        print("-"*80)

        # Predictions
        print(f"\nPREDICTIONS:")
        print(f"  Ensemble: {game['ENSEMBLE_PREDICTION']:+.1f} points")

        if game['ENSEMBLE_PREDICTION'] > 0:
            predicted_winner = game['HOME_TEAM']
            predicted_margin = game['ENSEMBLE_PREDICTION']
        else:
            predicted_winner = game['AWAY_TEAM']
            predicted_margin = abs(game['ENSEMBLE_PREDICTION'])

        print(f"  Predicted: {predicted_winner} by {predicted_margin:.1f}")

        # Actual results
        if pd.notna(game['home_score']) and pd.notna(game['away_score']):
            home_score = int(game['home_score'])
            away_score = int(game['away_score'])
            actual_margin = home_score - away_score

            print(f"\nACTUAL SCORE:")
            print(f"  {game['AWAY_TEAM']}: {away_score}")
            print(f"  {game['HOME_TEAM']}: {home_score}")
            print(f"  Margin: {abs(actual_margin):.0f} points")

            if actual_margin > 0:
                actual_winner = game['HOME_TEAM']
            elif actual_margin < 0:
                actual_winner = game['AWAY_TEAM']
            else:
                actual_winner = "TIE"

            print(f"  Winner: {actual_winner}")

            # Check if prediction was correct
            if "Final" in game['status']:
                completed += 1
                total_picks += 1

                prediction_correct = (
                    (game['ENSEMBLE_PREDICTION'] > 0 and actual_margin > 0) or
                    (game['ENSEMBLE_PREDICTION'] < 0 and actual_margin < 0)
                )

                if prediction_correct:
                    correct_picks += 1
                    print(f"\nRESULT: CORRECT")
                else:
                    print(f"\nRESULT: INCORRECT")

                # Error analysis
                prediction_error = abs(game['ENSEMBLE_PREDICTION'] - actual_margin)
                print(f"Margin Error: {prediction_error:.1f} points")

            else:
                in_progress += 1
                print(f"\nRESULT: IN PROGRESS")

        else:
            not_started += 1
            print(f"\nRESULT: NOT STARTED")

        print("="*80)
        print()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Games: {len(results)}")
    print(f"Completed: {completed}")
    print(f"In Progress: {in_progress}")
    print(f"Not Started: {not_started}")

    if total_picks > 0:
        accuracy = correct_picks / total_picks
        print(f"\nCURRENT ACCURACY: {correct_picks}/{total_picks} ({accuracy:.1%})")

        if accuracy >= 0.524:
            print(f"STATUS: PROFITABLE (need 52.4% for profitability)")
        else:
            print(f"STATUS: Below breakeven (need 52.4%)")

        # ROI calculation
        if total_picks > 0:
            bet_amount = 100
            win_return = correct_picks * (bet_amount + bet_amount * (100/110))
            total_spent = total_picks * bet_amount
            net_profit = win_return - total_spent
            roi = (net_profit / total_spent) * 100

            print(f"\nBETTING PERFORMANCE (${bet_amount}/bet):")
            print(f"  Total Risked: ${total_spent:,.2f}")
            print(f"  Net Profit: ${net_profit:+,.2f}")
            print(f"  ROI: {roi:+.2f}%")
    else:
        print(f"\nNo completed games yet - check back later!")

    print("="*80)

if __name__ == "__main__":
    compare_predictions()
