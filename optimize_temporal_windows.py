"""
Temporal Window Optimization via Grid Search

Tests different training window sizes, retraining frequencies, and walk-forward
strategies to find optimal configuration for dynamic NBA environment.

Key Questions:
1. What training window size performs best? (1yr, 2yr, 3yr, 4yr, 5yr)
2. Should we use expanding window (all history) or rolling window (recent N years)?
3. How often should we retrain? (Weekly, Monthly, Quarterly)
4. Does recency matter more than volume for NBA predictions?
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from models.elo_system import NBAEloRatings
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def train_elo_on_window(games_df, use_538=True, use_season_reset=True):
    """
    Train Elo system on given games window.

    Returns trained Elo system ready for predictions.
    """
    elo = NBAEloRatings(
        k_factor=20,
        home_advantage=100,
        initial_rating=1505,
        season_reset_factor=0.75,
        mean_rating=1505
    )

    if use_538:
        elo.margin_of_victory_multiplier = elo.margin_of_victory_multiplier_538

    games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)
    current_season = None

    for idx, row in games_df.iterrows():
        # Season reset if enabled
        if use_season_reset:
            game_date = pd.to_datetime(row['GAME_DATE'])
            game_season = game_date.year if game_date.month >= 10 else game_date.year - 1

            if current_season is not None and game_season != current_season:
                elo.season_reset()

            current_season = game_season

        # Update Elo ratings
        elo.update_ratings(
            row['TEAM_NAME_HOME'],
            row['TEAM_NAME_AWAY'],
            row['PTS_HOME'],
            row['PTS_AWAY'],
            is_home_a=True
        )

    return elo


def evaluate_predictions(predictions_df):
    """Evaluate prediction performance."""
    # Prediction accuracy
    predictions_df = predictions_df.copy()
    predictions_df['PREDICTED_WINNER'] = (predictions_df['EXPECTED_MARGIN'] > 0).astype(int)
    predictions_df['ACTUAL_WINNER'] = (predictions_df['ACTUAL_MARGIN'] > 0).astype(int)
    accuracy = (predictions_df['PREDICTED_WINNER'] == predictions_df['ACTUAL_WINNER']).mean()

    # Prediction error
    errors = predictions_df['ACTUAL_MARGIN'] - predictions_df['EXPECTED_MARGIN']
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())

    # Calibration (Brier score)
    brier = ((predictions_df['WIN_PROB'] - predictions_df['ACTUAL_WINNER']) ** 2).mean()

    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'brier': brier,
        'games': len(predictions_df)
    }


def walk_forward_validation(games_df, train_window_years, test_window_months=1,
                            window_type='rolling', step_months=1):
    """
    Walk-forward validation with configurable window size.

    Args:
        train_window_years: Years of training data (None = expanding window)
        test_window_months: Months of test data per step
        window_type: 'rolling' (fixed size) or 'expanding' (all history)
        step_months: How often to retrain

    Returns:
        Dictionary with performance metrics
    """
    games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

    min_date = games_df['GAME_DATE'].min()
    max_date = games_df['GAME_DATE'].max()

    # Start testing after minimum training period
    if train_window_years and window_type == 'rolling':
        test_start = min_date + pd.DateOffset(years=train_window_years)
    else:
        test_start = min_date + pd.DateOffset(years=2)  # Minimum 2 years for expanding

    all_predictions = []
    retrains = 0

    current_date = test_start

    while current_date < max_date:
        # Define training period
        if window_type == 'rolling' and train_window_years:
            train_start = current_date - pd.DateOffset(years=train_window_years)
            train_end = current_date
        else:  # expanding
            train_start = min_date
            train_end = current_date

        # Define test period
        test_end = current_date + pd.DateOffset(months=test_window_months)

        # Get data
        train_data = games_df[
            (games_df['GAME_DATE'] >= train_start) &
            (games_df['GAME_DATE'] < train_end)
        ]

        test_data = games_df[
            (games_df['GAME_DATE'] >= current_date) &
            (games_df['GAME_DATE'] < test_end)
        ]

        if len(test_data) == 0:
            break

        # Train Elo system
        elo = train_elo_on_window(train_data, use_538=True, use_season_reset=True)
        retrains += 1

        # Make predictions on test period
        for idx, row in test_data.iterrows():
            home_team = row['TEAM_NAME_HOME']
            away_team = row['TEAM_NAME_AWAY']

            # Predict
            win_prob, expected_margin = elo.predict_game(home_team, away_team, is_home_a=True)

            # Update for next game
            elo.update_ratings(
                home_team, away_team,
                row['PTS_HOME'], row['PTS_AWAY'],
                is_home_a=True
            )

            all_predictions.append({
                'GAME_DATE': row['GAME_DATE'],
                'HOME_TEAM': home_team,
                'AWAY_TEAM': away_team,
                'EXPECTED_MARGIN': expected_margin,
                'ACTUAL_MARGIN': row['PTS_HOME'] - row['PTS_AWAY'],
                'WIN_PROB': win_prob,
                'TRAIN_GAMES': len(train_data)
            })

        # Step forward
        current_date += pd.DateOffset(months=step_months)

    if len(all_predictions) == 0:
        return None

    predictions_df = pd.DataFrame(all_predictions)
    metrics = evaluate_predictions(predictions_df)
    metrics['retrains'] = retrains
    metrics['avg_train_size'] = predictions_df['TRAIN_GAMES'].mean()

    return metrics


def grid_search_temporal_windows(games_df):
    """
    Grid search over different temporal window configurations.

    Tests:
    - Window sizes: 1, 2, 3, 4, 5 years, expanding
    - Retraining frequencies: Monthly, Quarterly, Bi-annual
    """
    print("=" * 80)
    print("TEMPORAL WINDOW OPTIMIZATION - GRID SEARCH")
    print("=" * 80)

    # Configuration grid
    configurations = []

    # Rolling windows (fixed size)
    for years in [1, 2, 3, 4, 5]:
        for retrain_months in [1, 3, 6]:
            configurations.append({
                'window_type': 'rolling',
                'train_years': years,
                'retrain_months': retrain_months,
                'name': f'Rolling {years}yr, retrain {retrain_months}mo'
            })

    # Expanding windows (all history)
    for retrain_months in [1, 3, 6]:
        configurations.append({
            'window_type': 'expanding',
            'train_years': None,
            'retrain_months': retrain_months,
            'name': f'Expanding, retrain {retrain_months}mo'
        })

    results = []

    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Testing: {config['name']}")
        print("-" * 80)

        try:
            metrics = walk_forward_validation(
                games_df,
                train_window_years=config['train_years'],
                test_window_months=1,
                window_type=config['window_type'],
                step_months=config['retrain_months']
            )

            if metrics:
                print(f"  Accuracy: {metrics['accuracy']:.1%}")
                print(f"  MAE: {metrics['mae']:.3f} points")
                print(f"  Brier: {metrics['brier']:.4f}")
                print(f"  Games tested: {metrics['games']}")
                print(f"  Retrains: {metrics['retrains']}")
                print(f"  Avg train size: {metrics['avg_train_size']:.0f} games")

                results.append({
                    'configuration': config['name'],
                    'window_type': config['window_type'],
                    'train_years': config['train_years'],
                    'retrain_months': config['retrain_months'],
                    **metrics
                })
            else:
                print("  Insufficient data")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze grid search results and provide recommendations."""
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)

    # Sort by different metrics
    print("\n📊 TOP 5 BY ACCURACY:")
    top_accuracy = results_df.nlargest(5, 'accuracy')[
        ['configuration', 'accuracy', 'mae', 'brier', 'retrains']
    ]
    print(top_accuracy.to_string(index=False))

    print("\n📊 TOP 5 BY MAE (Lower is Better):")
    top_mae = results_df.nsmallest(5, 'mae')[
        ['configuration', 'accuracy', 'mae', 'brier', 'retrains']
    ]
    print(top_mae.to_string(index=False))

    print("\n📊 TOP 5 BY CALIBRATION (Brier, Lower is Better):")
    top_brier = results_df.nsmallest(5, 'brier')[
        ['configuration', 'accuracy', 'mae', 'brier', 'retrains']
    ]
    print(top_brier.to_string(index=False))

    # Analysis by window type
    print("\n" + "=" * 80)
    print("ANALYSIS BY WINDOW TYPE")
    print("=" * 80)

    rolling_results = results_df[results_df['window_type'] == 'rolling']
    expanding_results = results_df[results_df['window_type'] == 'expanding']

    print(f"\n🔄 ROLLING WINDOW (Recent N years only):")
    print(f"  Best accuracy: {rolling_results['accuracy'].max():.1%}")
    print(f"  Best MAE: {rolling_results['mae'].min():.3f}")
    print(f"  Avg accuracy: {rolling_results['accuracy'].mean():.1%}")

    print(f"\n📈 EXPANDING WINDOW (All history):")
    print(f"  Best accuracy: {expanding_results['accuracy'].max():.1%}")
    print(f"  Best MAE: {expanding_results['mae'].min():.3f}")
    print(f"  Avg accuracy: {expanding_results['accuracy'].mean():.1%}")

    # Analysis by retraining frequency
    print("\n" + "=" * 80)
    print("ANALYSIS BY RETRAINING FREQUENCY")
    print("=" * 80)

    for retrain_freq in [1, 3, 6]:
        freq_results = results_df[results_df['retrain_months'] == retrain_freq]
        print(f"\n⏱️  RETRAIN EVERY {retrain_freq} MONTH(S):")
        print(f"  Best accuracy: {freq_results['accuracy'].max():.1%}")
        print(f"  Best MAE: {freq_results['mae'].min():.3f}")
        print(f"  Avg retrains: {freq_results['retrains'].mean():.1f}")

    # Overall recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find best overall (balanced metrics)
    results_df['score'] = (
        results_df['accuracy'] * 0.4 +
        (1 - results_df['mae'] / results_df['mae'].max()) * 0.3 +
        (1 - results_df['brier'] / results_df['brier'].max()) * 0.3
    )

    best = results_df.loc[results_df['score'].idxmax()]

    print(f"\n🏆 BEST OVERALL CONFIGURATION:")
    print(f"  {best['configuration']}")
    print(f"  Accuracy: {best['accuracy']:.1%}")
    print(f"  MAE: {best['mae']:.3f} points")
    print(f"  Brier: {best['brier']:.4f}")
    print(f"  Retrains: {int(best['retrains'])}")
    print(f"  Avg training size: {best['avg_train_size']:.0f} games")

    # Practical recommendation
    print(f"\n💡 PRACTICAL RECOMMENDATION:")

    # Balance performance vs. computational cost
    good_configs = results_df[
        (results_df['accuracy'] >= results_df['accuracy'].quantile(0.75)) &
        (results_df['mae'] <= results_df['mae'].quantile(0.25))
    ].sort_values('retrains')

    if len(good_configs) > 0:
        practical = good_configs.iloc[0]
        print(f"  {practical['configuration']}")
        print(f"  Accuracy: {practical['accuracy']:.1%}")
        print(f"  MAE: {practical['mae']:.3f} points")
        print(f"  Retrains needed: {int(practical['retrains'])}")
        print(f"\n  Rationale:")
        print(f"  - High performance (top 25% accuracy, top 25% MAE)")
        print(f"  - Fewer retrains = less computational cost")
        print(f"  - Good balance of recency and stability")

    return results_df


def main():
    """Run temporal window optimization."""
    print("Loading data...")
    games = pd.read_csv('data/raw/games_processed.csv')
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE_HOME'])

    # Use data from 2019 onwards for faster testing
    # (Can use full dataset for final optimization)
    games = games[games['GAME_DATE'] >= '2019-01-01'].copy()

    print(f"Data loaded: {len(games)} games from {games['GAME_DATE'].min().date()} to {games['GAME_DATE'].max().date()}")

    # Run grid search
    results = grid_search_temporal_windows(games)

    # Save results
    results.to_csv('results/temporal_window_optimization.csv', index=False)
    print(f"\n✅ Results saved to: results/temporal_window_optimization.csv")

    # Analyze and recommend
    results = analyze_results(results)

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
