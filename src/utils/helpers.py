"""
Helper utilities for NBA prediction model
"""

import pandas as pd
import numpy as np
import yaml
import os

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Evaluate prediction performance

    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    - y_prob: Prediction probabilities (optional)

    Returns:
    - Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

    return metrics

def evaluate_spread_predictions(actual_margin, predicted_margin, threshold=0):
    """
    Evaluate point spread prediction accuracy

    Parameters:
    - actual_margin: Actual point differential
    - predicted_margin: Predicted point differential
    - threshold: Spread line (default 0 for straight up)

    Returns:
    - Dictionary with spread metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Against the spread accuracy
    actual_cover = (actual_margin > threshold).astype(int)
    predicted_cover = (predicted_margin > threshold).astype(int)
    ats_accuracy = (actual_cover == predicted_cover).mean()

    # Regression metrics
    mae = mean_absolute_error(actual_margin, predicted_margin)
    rmse = np.sqrt(mean_squared_error(actual_margin, predicted_margin))

    # Correlation
    correlation = np.corrcoef(actual_margin, predicted_margin)[0, 1]

    return {
        'ats_accuracy': ats_accuracy,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation
    }

def calculate_betting_roi(predictions_df, bet_amount=100):
    """
    Calculate ROI assuming -110 odds on all bets

    Parameters:
    - predictions_df: DataFrame with 'actual_cover' and 'predicted_cover' columns
    - bet_amount: Amount bet per game

    Returns:
    - Dictionary with betting performance metrics
    """
    correct_bets = (predictions_df['actual_cover'] == predictions_df['predicted_cover']).sum()
    total_bets = len(predictions_df)
    win_rate = correct_bets / total_bets

    # -110 odds = risk $110 to win $100
    # If betting $100 per game:
    # - Win: get back $100 (stake) + $90.91 (profit) = $190.91
    # - Loss: lose $100
    incorrect_bets = total_bets - correct_bets

    # Calculate net profit/loss
    win_return = correct_bets * (bet_amount + bet_amount * (100/110))  # Stake + winnings
    loss_amount = incorrect_bets * bet_amount  # Lose the stake
    total_spent = total_bets * bet_amount

    net_profit = win_return - total_spent
    roi = (net_profit / total_spent) * 100 if total_spent > 0 else 0

    return {
        'total_bets': total_bets,
        'correct_bets': correct_bets,
        'win_rate': win_rate,
        'total_risked': total_spent,
        'profit': net_profit,
        'roi_percent': roi,
        'breakeven_rate': 52.38  # Need to win 52.38% to break even at -110
    }

def create_train_test_split(df, test_season='2023-24', date_col='GAME_DATE'):
    """
    Split data into train/test by season
    """
    df[date_col] = pd.to_datetime(df[date_col])

    # Assuming season format in dataframe
    train = df[df['SEASON'] != test_season].copy()
    test = df[df['SEASON'] == test_season].copy()

    print(f"Training set: {len(train)} games")
    print(f"Test set: {len(test)} games")

    return train, test

def save_results(results, output_path):
    """Save results to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def print_metrics(metrics, title="Model Performance"):
    """Pretty print metrics"""
    print("\n" + "="*50)
    print(title)
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print("="*50)
