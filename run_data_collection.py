"""
Main script to run data collection pipeline
"""

import sys
import os
sys.path.append('src')

from data_collection.nba_scraper import NBADataScraper
from data_collection.download_538_data import download_538_elo_data
from feature_engineering.feature_builder import NBAFeatureBuilder
from models.elo_system import NBAEloRatings
import pandas as pd

def main():
    print("\n" + "="*60)
    print(" NBA PREDICTION MODEL - DATA COLLECTION PIPELINE")
    print("="*60)

    # Step 1: Scrape NBA game data
    print("\n[1/4] Scraping NBA game data from nba_api...")
    scraper = NBADataScraper(output_dir='data/raw')

    seasons = [
        '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24'
    ]

    games_df = scraper.scrape_multiple_seasons(seasons)

    if games_df is None:
        print("ERROR: Failed to scrape game data")
        return

    # Step 2: Download FiveThirtyEight data
    print("\n[2/4] Downloading FiveThirtyEight historical Elo data...")
    elo_538 = download_538_elo_data(output_dir='data/raw')

    # Rename columns for consistency
    if 'GAME_DATE_HOME' in games_df.columns and 'GAME_DATE' not in games_df.columns:
        games_df['GAME_DATE'] = games_df['GAME_DATE_HOME']

    # Add SEASON column based on date if not present
    if 'SEASON' not in games_df.columns:
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        games_df['SEASON'] = games_df['GAME_DATE'].apply(
            lambda x: f"{x.year-1}-{str(x.year)[-2:]}" if x.month < 10 else f"{x.year}-{str(x.year+1)[-2:]}"
        )

    # Step 3: Build features
    print("\n[3/4] Building features...")
    feature_builder = NBAFeatureBuilder(rolling_windows=[5, 10, 20])
    features = feature_builder.build_features(games_df)

    # Save feature sets
    features['rest'].to_csv('data/features/rest_days.csv', index=False)
    features['rolling'].to_csv('data/features/rolling_stats.csv', index=False)
    features['momentum'].to_csv('data/features/momentum.csv', index=False)
    features['h2h'].to_csv('data/features/head_to_head.csv', index=False)

    print("\nFeatures saved to data/features/")

    # Step 4: Calculate Elo ratings
    print("\n[4/4] Calculating Elo ratings...")
    elo_system = NBAEloRatings(k_factor=20, home_advantage=100)

    # Make sure required columns exist
    if 'GAME_DATE' not in games_df.columns:
        # Use GAME_DATE_HOME if available
        games_df['GAME_DATE'] = games_df['GAME_DATE_HOME'] if 'GAME_DATE_HOME' in games_df.columns else games_df.index

    elo_results = elo_system.process_season(games_df)
    elo_results.to_csv('data/features/elo_ratings.csv', index=False)

    print("\nElo ratings saved to data/features/elo_ratings.csv")

    # Current standings
    current_ratings = elo_system.get_current_ratings()
    print("\nCurrent Elo Ratings (Top 10):")
    print(current_ratings.head(10))

    # Summary statistics
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Total games collected: {len(games_df)}")
    print(f"Seasons covered: {', '.join(seasons)}")
    print(f"Teams tracked: {len(current_ratings)}")
    print(f"\nFeature sets created:")
    print(f"  - Rest days: {len(features['rest'])} records")
    print(f"  - Rolling stats: {len(features['rolling'])} records")
    print(f"  - Momentum: {len(features['momentum'])} records")
    print(f"  - Head-to-head: {len(features['h2h'])} records")
    print(f"  - Elo ratings: {len(elo_results)} records")
    print("="*60)

    print("\nData collection complete!")
    print("Next step: Run model training (run_model_training.py)")

if __name__ == "__main__":
    main()
