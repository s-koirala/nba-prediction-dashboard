"""
NBA Data Scraper using nba_api
Collects game data, team statistics, and player information
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, leaguedashteamstats
from nba_api.stats.static import teams
from datetime import datetime
import time
import os
from tqdm import tqdm

class NBADataScraper:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        self.teams = teams.get_teams()
        os.makedirs(output_dir, exist_ok=True)

    def get_season_games(self, season='2023-24', league_id='00'):
        """
        Fetch all games for a given season
        season: Format '2023-24'
        league_id: '00' for NBA
        """
        print(f"Fetching games for {season} season...")

        try:
            # Use LeagueGameFinder to get all games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable=league_id,
                season_type_nullable='Regular Season'
            )
            games = gamefinder.get_data_frames()[0]

            # Each game appears twice (once for each team), so we need to deduplicate
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

            print(f"Found {len(games)} team-game records ({len(games)//2} unique games)")
            return games

        except Exception as e:
            print(f"Error fetching season games: {e}")
            return None

    def get_team_stats(self, season='2023-24'):
        """
        Fetch team-level statistics for the season
        """
        print(f"Fetching team stats for {season}...")

        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_nullable='Regular Season'
            )
            df = team_stats.get_data_frames()[0]
            print(f"Fetched stats for {len(df)} teams")
            return df

        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return None

    def process_game_data(self, games_df):
        """
        Process raw game data to create structured dataset with home/away teams
        """
        print("Processing game data...")

        # Split into home and away
        games_df['IS_HOME'] = games_df['MATCHUP'].str.contains('vs.')

        # Create unique game identifier
        games_df = games_df.sort_values(['GAME_DATE', 'GAME_ID', 'IS_HOME'], ascending=[True, True, False])

        # Separate home and away games
        home_games = games_df[games_df['IS_HOME'] == True].copy()
        away_games = games_df[games_df['IS_HOME'] == False].copy()

        # Merge to create one row per game
        home_games = home_games.add_suffix('_HOME')
        away_games = away_games.add_suffix('_AWAY')

        merged = pd.merge(
            home_games,
            away_games,
            left_on='GAME_ID_HOME',
            right_on='GAME_ID_AWAY',
            how='inner'
        )

        # Create target variables
        merged['HOME_WIN'] = (merged['WL_HOME'] == 'W').astype(int)
        merged['POINT_DIFFERENTIAL'] = merged['PTS_HOME'] - merged['PTS_AWAY']  # Positive = home team won by X
        merged['TOTAL_POINTS'] = merged['PTS_HOME'] + merged['PTS_AWAY']

        # Select key columns
        key_cols = [
            'GAME_ID_HOME', 'GAME_DATE_HOME',
            'TEAM_NAME_HOME', 'TEAM_NAME_AWAY',
            'PTS_HOME', 'PTS_AWAY',
            'FG_PCT_HOME', 'FG_PCT_AWAY',
            'FG3_PCT_HOME', 'FG3_PCT_AWAY',
            'FT_PCT_HOME', 'FT_PCT_AWAY',
            'REB_HOME', 'REB_AWAY',
            'AST_HOME', 'AST_AWAY',
            'STL_HOME', 'STL_AWAY',
            'BLK_HOME', 'BLK_AWAY',
            'TOV_HOME', 'TOV_AWAY',
            'HOME_WIN', 'POINT_DIFFERENTIAL', 'TOTAL_POINTS'
        ]

        processed = merged[key_cols].copy()
        processed.columns = [col.replace('_HOME', '').replace('_AWAY', '_AWAY')
                            if '_AWAY' in col else col.replace('_HOME', '_HOME')
                            for col in processed.columns]

        print(f"Processed {len(processed)} games")
        return processed

    def scrape_multiple_seasons(self, seasons):
        """
        Scrape data for multiple seasons
        seasons: List of season strings like ['2020-21', '2021-22']
        """
        all_games = []

        for season in tqdm(seasons, desc="Scraping seasons"):
            print(f"\n{'='*50}")
            print(f"Season: {season}")
            print(f"{'='*50}")

            # Get games
            games = self.get_season_games(season)
            if games is not None:
                games['SEASON'] = season
                all_games.append(games)

            # Be nice to the API
            time.sleep(1)

        # Combine all seasons
        if all_games:
            combined = pd.concat(all_games, ignore_index=True)

            # Process the data
            processed = self.process_game_data(combined)

            # Save raw data
            raw_path = os.path.join(self.output_dir, 'games_raw.csv')
            combined.to_csv(raw_path, index=False)
            print(f"\nSaved raw data to {raw_path}")

            # Save processed data
            processed_path = os.path.join(self.output_dir, 'games_processed.csv')
            processed.to_csv(processed_path, index=False)
            print(f"Saved processed data to {processed_path}")

            return processed

        return None

if __name__ == "__main__":
    # Initialize scraper
    scraper = NBADataScraper(output_dir='../../data/raw')

    # Define seasons to scrape (last 5 seasons)
    seasons = [
        '2019-20', '2020-21', '2021-22', '2022-23', '2023-24'
    ]

    # Scrape data
    print("Starting NBA data scraping...")
    print(f"Seasons: {', '.join(seasons)}")

    data = scraper.scrape_multiple_seasons(seasons)

    if data is not None:
        print(f"\n{'='*50}")
        print("SCRAPING COMPLETE")
        print(f"{'='*50}")
        print(f"Total games collected: {len(data)}")
        print(f"\nData preview:")
        print(data.head())
        print(f"\nData shape: {data.shape}")
        print(f"\nColumns: {list(data.columns)}")
