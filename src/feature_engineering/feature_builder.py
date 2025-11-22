"""
Feature Engineering for NBA Prediction Model
Creates rolling averages, momentum indicators, and advanced metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class NBAFeatureBuilder:
    def __init__(self, rolling_windows=[5, 10, 20]):
        self.rolling_windows = rolling_windows

    def calculate_rest_days(self, games_df):
        """
        Calculate days of rest between games for each team
        """
        print("Calculating rest days...")

        games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

        # Create team-game records
        home_games = games_df[['GAME_DATE', 'TEAM_NAME_HOME']].copy()
        home_games.columns = ['GAME_DATE', 'TEAM']
        home_games['IS_HOME'] = 1

        away_games = games_df[['GAME_DATE', 'TEAM_NAME_AWAY']].copy()
        away_games.columns = ['GAME_DATE', 'TEAM']
        away_games['IS_HOME'] = 0

        team_games = pd.concat([home_games, away_games]).sort_values(['TEAM', 'GAME_DATE'])

        # Calculate rest days
        team_games['PREV_GAME_DATE'] = team_games.groupby('TEAM')['GAME_DATE'].shift(1)
        team_games['REST_DAYS'] = (team_games['GAME_DATE'] - team_games['PREV_GAME_DATE']).dt.days

        # Fill first game of season with average rest (2 days)
        team_games['REST_DAYS'] = team_games['REST_DAYS'].fillna(2)

        # Back-to-back indicator
        team_games['IS_BACK_TO_BACK'] = (team_games['REST_DAYS'] <= 1).astype(int)

        return team_games[['GAME_DATE', 'TEAM', 'REST_DAYS', 'IS_BACK_TO_BACK']]

    def calculate_rolling_stats(self, games_df, windows=[5, 10, 20]):
        """
        Calculate rolling averages for key statistics
        """
        print(f"Calculating rolling averages for windows: {windows}...")

        # Create long format for easier rolling calculations
        home_stats = games_df[[
            'GAME_DATE', 'TEAM_NAME_HOME', 'PTS_HOME', 'FG_PCT_HOME',
            'FG3_PCT_HOME', 'FT_PCT_HOME', 'REB_HOME', 'AST_HOME',
            'STL_HOME', 'BLK_HOME', 'TOV_HOME'
        ]].copy()
        home_stats.columns = [c.replace('_HOME', '') for c in home_stats.columns]
        home_stats['IS_HOME'] = 1

        away_stats = games_df[[
            'GAME_DATE', 'TEAM_NAME_AWAY', 'PTS_AWAY', 'FG_PCT_AWAY',
            'FG3_PCT_AWAY', 'FT_PCT_AWAY', 'REB_AWAY', 'AST_AWAY',
            'STL_AWAY', 'BLK_AWAY', 'TOV_AWAY'
        ]].copy()
        away_stats.columns = [c.replace('_AWAY', '') for c in away_stats.columns]
        away_stats['IS_HOME'] = 0

        all_stats = pd.concat([home_stats, away_stats]).sort_values(['TEAM_NAME', 'GAME_DATE'])

        # Calculate rolling averages
        stat_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']

        for window in windows:
            for col in stat_cols:
                # Use shift(1) to avoid lookahead bias (don't include current game)
                all_stats[f'{col}_ROLL_{window}'] = (
                    all_stats.groupby('TEAM_NAME')[col]
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        return all_stats

    def calculate_momentum_features(self, games_df):
        """
        Calculate win/loss streaks and recent form
        """
        print("Calculating momentum features...")

        # Create win/loss records
        home_results = games_df[['GAME_DATE', 'TEAM_NAME_HOME', 'HOME_WIN']].copy()
        home_results.columns = ['GAME_DATE', 'TEAM', 'WON']
        home_results['IS_HOME'] = 1

        away_results = games_df[['GAME_DATE', 'TEAM_NAME_AWAY', 'HOME_WIN']].copy()
        away_results['WON'] = 1 - away_results['HOME_WIN']
        away_results.columns = ['GAME_DATE', 'TEAM', 'WON', 'IS_HOME']
        away_results['IS_HOME'] = 0

        results = pd.concat([home_results, away_results]).sort_values(['TEAM', 'GAME_DATE'])

        # Win percentage last N games
        for window in [5, 10]:
            results[f'WIN_PCT_L{window}'] = (
                results.groupby('TEAM')['WON']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Current win streak (positive) or loss streak (negative)
        # IMPORTANT: Streak should be from BEFORE the current game (no leakage!)
        results['STREAK'] = 0
        for team in results['TEAM'].unique():
            team_mask = results['TEAM'] == team
            team_data = results[team_mask]['WON'].values

            streak = 0
            streaks = []
            for won in team_data:
                # Append current streak BEFORE updating (pre-game value)
                streaks.append(streak)
                # Then update streak based on this game's result
                if won == 1:
                    streak = streak + 1 if streak >= 0 else 1
                else:
                    streak = streak - 1 if streak <= 0 else -1

            results.loc[team_mask, 'STREAK'] = streaks

        return results[['GAME_DATE', 'TEAM', 'WIN_PCT_L5', 'WIN_PCT_L10', 'STREAK']]

    def calculate_matchup_history(self, games_df):
        """
        Calculate head-to-head historical performance
        """
        print("Calculating matchup history...")

        games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)

        # Create matchup key (alphabetically sorted teams)
        games_df['MATCHUP_KEY'] = games_df.apply(
            lambda row: '_vs_'.join(sorted([row['TEAM_NAME_HOME'], row['TEAM_NAME_AWAY']])),
            axis=1
        )

        # Track head-to-head record
        h2h_records = {}
        h2h_stats = []

        for idx, row in games_df.iterrows():
            matchup_key = row['MATCHUP_KEY']
            home_team = row['TEAM_NAME_HOME']
            away_team = row['TEAM_NAME_AWAY']

            # Get historical record for this matchup
            if matchup_key not in h2h_records:
                h2h_records[matchup_key] = {
                    'games': 0,
                    home_team: 0,
                    away_team: 0,
                    'total_margin': 0
                }

            record = h2h_records[matchup_key]

            # Calculate features from history
            games_played = record['games']
            home_wins = record.get(home_team, 0)
            avg_margin = record['total_margin'] / games_played if games_played > 0 else 0

            h2h_stats.append({
                'GAME_DATE': row['GAME_DATE'],
                'HOME_TEAM': home_team,
                'AWAY_TEAM': away_team,
                'H2H_GAMES': games_played,
                'H2H_HOME_WINS': home_wins,
                'H2H_AVG_MARGIN': avg_margin
            })

            # Update record after this game
            record['games'] += 1
            winner = home_team if row['HOME_WIN'] == 1 else away_team
            record[winner] = record.get(winner, 0) + 1
            record['total_margin'] += row['POINT_DIFFERENTIAL']

        return pd.DataFrame(h2h_stats)

    def build_features(self, games_df):
        """
        Main method to build all features
        """
        print("\n" + "="*50)
        print("BUILDING FEATURES")
        print("="*50)

        games_df = games_df.copy()
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

        # 1. Rest days
        rest_features = self.calculate_rest_days(games_df)

        # 2. Rolling statistics
        rolling_features = self.calculate_rolling_stats(games_df, windows=self.rolling_windows)

        # 3. Momentum features
        momentum_features = self.calculate_momentum_features(games_df)

        # 4. Head-to-head history
        h2h_features = self.calculate_matchup_history(games_df)

        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*50)

        return {
            'rest': rest_features,
            'rolling': rolling_features,
            'momentum': momentum_features,
            'h2h': h2h_features
        }

if __name__ == "__main__":
    print("Feature Builder initialized")
    print("Use this module to create features for NBA prediction model")
