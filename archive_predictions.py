"""
Archive today's predictions before generating new ones.
This preserves the original predictions made BEFORE games were played.
"""

import pandas as pd
import os
from datetime import datetime

def archive_todays_predictions():
    """Save tonight's predictions with today's date for later historical tracking"""

    if not os.path.exists('results/tonights_predictions.csv'):
        print("No predictions file to archive")
        return

    try:
        # Load tonight's predictions
        predictions = pd.read_csv('results/tonights_predictions.csv')

        if len(predictions) == 0:
            print("Empty predictions file, nothing to archive")
            return

        # Create archive directory
        os.makedirs('results/prediction_archive', exist_ok=True)

        # Save with today's date
        today = datetime.now().strftime('%Y-%m-%d')
        archive_path = f'results/prediction_archive/predictions_{today}.csv'

        predictions.to_csv(archive_path, index=False)
        print(f"✅ Archived {len(predictions)} predictions to {archive_path}")

    except Exception as e:
        print(f"Error archiving predictions: {e}")

if __name__ == "__main__":
    archive_todays_predictions()
