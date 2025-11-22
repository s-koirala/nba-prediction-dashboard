"""
Download FiveThirtyEight's historical NBA Elo data
"""

import requests
import pandas as pd
import os

def download_538_elo_data(output_dir='data/raw'):
    """
    Download FiveThirtyEight's historical NBA Elo ratings
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading FiveThirtyEight NBA Elo data...")

    # URL to FiveThirtyEight's NBA Elo dataset
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"

    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()

        # Save to file
        output_path = os.path.join(output_dir, 'fivethirtyeight_elo.csv')
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"Successfully downloaded to {output_path}")

        # Load and preview the data
        df = pd.read_csv(output_path)
        print(f"\nDataset shape: {df.shape}")

        # Check if date column exists (may be named differently)
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

        if date_col:
            print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

        print(f"\nColumns: {list(df.columns)}")
        print(f"\nPreview:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error downloading data: {e}")
        # Don't fail the entire pipeline if 538 data fails
        print("Continuing without FiveThirtyEight data...")
        return None

def download_538_forecasts(output_dir='data/raw'):
    """
    Download FiveThirtyEight's NBA forecasts (if available)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nDownloading FiveThirtyEight NBA forecasts...")

    # URL to forecasts (note: these may be outdated as of 2023)
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-forecasts/nba_elo_latest.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()

        output_path = os.path.join(output_dir, 'fivethirtyeight_forecasts.csv')
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"Successfully downloaded to {output_path}")

        df = pd.read_csv(output_path)
        print(f"Forecast records: {len(df)}")
        print(f"\nPreview:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error downloading forecasts: {e}")
        print("Note: FiveThirtyEight stopped updating NBA forecasts in June 2023")
        return None

if __name__ == "__main__":
    print("Downloading FiveThirtyEight data...\n")

    # Download Elo historical data
    elo_data = download_538_elo_data(output_dir='../../data/raw')

    # Try to download forecasts (may not be current)
    forecast_data = download_538_forecasts(output_dir='../../data/raw')

    print("\n" + "="*50)
    print("Download complete!")
    print("="*50)
