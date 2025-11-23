"""
Elo Rating System for NBA
Based on FiveThirtyEight's methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

class NBAEloRatings:
    def __init__(self, k_factor=20, home_advantage=100, initial_rating=1505,
                 season_reset_factor=0.75, mean_rating=1505):
        """
        Initialize Elo rating system

        Parameters:
        - k_factor: How quickly ratings change (higher = more volatile)
        - home_advantage: Points added to home team's rating
        - initial_rating: Starting Elo rating for new teams
        - season_reset_factor: Regression factor for season reset (FiveThirtyEight uses 0.75)
        - mean_rating: Mean rating to regress towards (FiveThirtyEight uses 1505)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.season_reset_factor = season_reset_factor
        self.mean_rating = mean_rating
        self.ratings = {}  # Dictionary to store current ratings

    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected probability of team A winning
        Uses standard Elo formula
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def margin_of_victory_multiplier(self, point_diff, elo_diff):
        """
        Adjust K-factor based on margin of victory
        Larger wins matter more, but with diminishing returns
        """
        # FiveThirtyEight-style MOV multiplier
        mov = abs(point_diff)

        # Base multiplier on MOV
        if mov <= 3:
            multiplier = 1.0
        else:
            # Logarithmic scaling for MOV
            multiplier = np.log(mov + 1) / np.log(4)

        # Adjust for expected result (upsets matter more)
        if elo_diff > 0 and point_diff < 0:
            # Underdog won - increase multiplier
            multiplier *= 1.5
        elif elo_diff < 0 and point_diff > 0:
            # Favorite won as expected - decrease slightly
            multiplier *= 0.9

        return multiplier

    def update_ratings(self, team_a, team_b, score_a, score_b, is_home_a=True):
        """
        Update Elo ratings after a game

        Parameters:
        - team_a: Name of team A
        - team_b: Name of team B
        - score_a: Points scored by team A
        - score_b: Points scored by team B
        - is_home_a: Whether team A is home team

        Returns:
        - Tuple of (new_rating_a, new_rating_b, expected_win_prob_a)
        """
        # Get current ratings (or initialize if new team)
        rating_a = self.ratings.get(team_a, self.initial_rating)
        rating_b = self.ratings.get(team_b, self.initial_rating)

        # Apply home court advantage
        if is_home_a:
            rating_a_adjusted = rating_a + self.home_advantage
            rating_b_adjusted = rating_b
        else:
            rating_a_adjusted = rating_a
            rating_b_adjusted = rating_b + self.home_advantage

        # Calculate expected score
        expected_a = self.expected_score(rating_a_adjusted, rating_b_adjusted)

        # Actual result (1 for win, 0 for loss)
        actual_a = 1 if score_a > score_b else 0

        # Calculate margin of victory
        point_diff = score_a - score_b
        elo_diff = rating_a_adjusted - rating_b_adjusted

        # MOV multiplier
        mov_mult = self.margin_of_victory_multiplier(point_diff, elo_diff)

        # Update ratings
        rating_change = self.k_factor * mov_mult * (actual_a - expected_a)

        new_rating_a = rating_a + rating_change
        new_rating_b = rating_b - rating_change

        # Store updated ratings
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b

        return new_rating_a, new_rating_b, expected_a

    def margin_of_victory_multiplier_538(self, point_diff, elo_diff):
        """
        FiveThirtyEight's exact MOV multiplier formula
        K_multiplier = (MOV + 3)^0.8 / (7.5 + 0.006 * ED)

        This is the official FiveThirtyEight formula.
        """
        mov = abs(point_diff)
        ed = abs(elo_diff)

        # FiveThirtyEight's formula
        k_multiplier = ((mov + 3) ** 0.8) / (7.5 + 0.006 * ed)

        return k_multiplier

    def season_reset(self):
        """
        Apply season reset - regress ratings toward mean.
        FiveThirtyEight methodology: new_rating = (0.75 * current) + (0.25 * 1505)

        This should be called at the start of each new season.
        """
        for team in self.ratings:
            current_rating = self.ratings[team]
            new_rating = (self.season_reset_factor * current_rating) + \
                        ((1 - self.season_reset_factor) * self.mean_rating)
            self.ratings[team] = new_rating

        print(f"Season reset applied: {self.season_reset_factor:.0%} retention toward mean of {self.mean_rating}")

    def predict_game(self, team_a, team_b, is_home_a=True):
        """
        Predict outcome of a game

        Returns:
        - win_probability_a: Probability team A wins
        - expected_margin: Expected point differential (positive = A wins)
        """
        rating_a = self.ratings.get(team_a, self.initial_rating)
        rating_b = self.ratings.get(team_b, self.initial_rating)

        # Apply home court advantage
        if is_home_a:
            rating_a_adjusted = rating_a + self.home_advantage
            rating_b_adjusted = rating_b
        else:
            rating_a_adjusted = rating_a
            rating_b_adjusted = rating_b + self.home_advantage

        # Win probability
        win_prob_a = self.expected_score(rating_a_adjusted, rating_b_adjusted)

        # Expected margin (approximate)
        # Rule of thumb: 25 Elo points ≈ 1 point in score
        elo_diff = rating_a_adjusted - rating_b_adjusted
        expected_margin = elo_diff / 25

        return win_prob_a, expected_margin

    def process_season(self, games_df):
        """
        Process entire season and calculate Elo ratings for each game

        Parameters:
        - games_df: DataFrame with columns ['GAME_DATE', 'TEAM_NAME_HOME', 'TEAM_NAME_AWAY', 'PTS_HOME', 'PTS_AWAY']

        Returns:
        - DataFrame with Elo ratings and predictions added
        """
        results = []

        # Sort by date
        games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)

        for idx, row in games_df.iterrows():
            home_team = row['TEAM_NAME_HOME']
            away_team = row['TEAM_NAME_AWAY']
            home_score = row['PTS_HOME']
            away_score = row['PTS_AWAY']

            # Get predictions BEFORE updating ratings
            win_prob_home, expected_margin = self.predict_game(home_team, away_team, is_home_a=True)

            # Get pre-game ratings
            pre_rating_home = self.ratings.get(home_team, self.initial_rating)
            pre_rating_away = self.ratings.get(away_team, self.initial_rating)

            # Update ratings
            post_rating_home, post_rating_away, _ = self.update_ratings(
                home_team, away_team, home_score, away_score, is_home_a=True
            )

            # Store results
            results.append({
                'GAME_DATE': row['GAME_DATE'],
                'HOME_TEAM': home_team,
                'AWAY_TEAM': away_team,
                'HOME_SCORE': home_score,
                'AWAY_SCORE': away_score,
                'PRE_ELO_HOME': pre_rating_home,
                'PRE_ELO_AWAY': pre_rating_away,
                'POST_ELO_HOME': post_rating_home,
                'POST_ELO_AWAY': post_rating_away,
                'WIN_PROB_HOME': win_prob_home,
                'EXPECTED_MARGIN': expected_margin,
                'ACTUAL_MARGIN': home_score - away_score
            })

        return pd.DataFrame(results)

    def get_current_ratings(self):
        """Return current Elo ratings for all teams"""
        return pd.DataFrame(
            list(self.ratings.items()),
            columns=['TEAM', 'ELO_RATING']
        ).sort_values('ELO_RATING', ascending=False)

if __name__ == "__main__":
    print("Elo Rating System initialized")
    print("Use this module to calculate Elo ratings for NBA teams")
