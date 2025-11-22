"""
Optimize bet sizing for different confidence levels using grid search
"""

import pandas as pd
import numpy as np
from itertools import product
import json

def get_confidence_level(predictions_row):
    """Calculate confidence level based on model agreement"""
    preds = [
        predictions_row['elo_pred'],
        predictions_row['nn_pred'],
        predictions_row['xgb_pred'],
        predictions_row['ensemble_pred']
    ]

    spread = max(preds) - min(preds)

    if spread < 3:
        return "HIGH"
    elif spread < 6:
        return "MEDIUM"
    else:
        return "LOW"

def simulate_betting_strategy(historical_df, high_pct, medium_pct, low_pct, initial_bankroll=10000):
    """
    Simulate betting with FIXED percentages of initial bankroll for each confidence level
    (No compounding - bet sizes stay constant based on initial bankroll)
    Returns ROI and other metrics
    """
    total_wagered = 0
    total_profit = 0
    num_bets = 0
    num_wins = 0

    for idx, row in historical_df.iterrows():
        confidence = row['confidence']

        # Determine bet size as FIXED percentage of initial bankroll
        if confidence == "HIGH":
            bet_pct = high_pct
        elif confidence == "MEDIUM":
            bet_pct = medium_pct
        else:
            bet_pct = low_pct

        bet_amount = initial_bankroll * bet_pct
        total_wagered += bet_amount
        num_bets += 1

        # Check if prediction was correct
        actual_correct = (row['ensemble_pred'] > 0) == (row['actual_margin'] > 0)

        if actual_correct:
            # Win: profit at -110 odds
            profit = bet_amount * (100/110)
            total_profit += profit
            num_wins += 1
        else:
            # Loss: lose the bet
            total_profit -= bet_amount

    # Calculate ROI based on total amount wagered
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    win_rate = num_wins / num_bets if num_bets > 0 else 0

    return {
        'total_profit': total_profit,
        'total_wagered': total_wagered,
        'roi': roi,
        'win_rate': win_rate,
        'num_bets': num_bets,
        'num_wins': num_wins,
        'high_pct': high_pct,
        'medium_pct': medium_pct,
        'low_pct': low_pct
    }

def grid_search_optimal_bets(historical_df, resolution=0.005):
    """
    Grid search to find optimal bet percentages for each confidence level

    Search space:
    - HIGH: 1% to 10%
    - MEDIUM: 0.5% to 8%
    - LOW: 0% to 5%
    """
    print("\n" + "="*80)
    print("GRID SEARCH FOR OPTIMAL BET SIZING")
    print("="*80)

    print(f"\nSearching with resolution: {resolution*100:.1f}%")
    print(f"Total historical games: {len(historical_df)}")

    # Confidence distribution
    conf_counts = historical_df['confidence'].value_counts()
    print(f"\nConfidence Distribution:")
    for conf, count in conf_counts.items():
        pct = count / len(historical_df) * 100
        win_rate = sum((historical_df[historical_df['confidence']==conf]['correct'])) / count * 100
        print(f"  {conf}: {count} games ({pct:.1f}%) - Win Rate: {win_rate:.1f}%")

    # Define search ranges (allow wider ranges since LOW has best performance)
    high_range = np.arange(0.0, 0.08, resolution)
    medium_range = np.arange(0.0, 0.08, resolution)
    low_range = np.arange(0.0, 0.10, resolution)

    total_combinations = len(high_range) * len(medium_range) * len(low_range)
    print(f"\nTesting {total_combinations:,} combinations...")

    best_roi = -float('inf')
    best_params = None
    all_results = []

    tested = 0
    for high_pct in high_range:
        for medium_pct in medium_range:
            for low_pct in low_range:
                # No constraints - let optimizer find best allocation based on actual performance
                result = simulate_betting_strategy(
                    historical_df, high_pct, medium_pct, low_pct
                )

                all_results.append(result)

                if result['roi'] > best_roi:
                    best_roi = result['roi']
                    best_params = result

                tested += 1

    print(f"Tested {tested:,} valid combinations")

    print("\n" + "="*80)
    print("OPTIMAL BET SIZING")
    print("="*80)
    print(f"\nHIGH Confidence: {best_params['high_pct']*100:.2f}% of bankroll")
    print(f"MEDIUM Confidence: {best_params['medium_pct']*100:.2f}% of bankroll")
    print(f"LOW Confidence: {best_params['low_pct']*100:.2f}% of bankroll")
    print(f"\nExpected ROI: {best_params['roi']:.2f}%")
    print(f"Total Profit: ${best_params['total_profit']:,.2f}")
    print(f"Total Wagered: ${best_params['total_wagered']:,.2f}")
    print(f"Historical Win Rate: {best_params['win_rate']:.2%}")
    print(f"Total Bets: {best_params['num_bets']} ({best_params['num_wins']} wins)")

    # Show top 10 strategies
    print("\n" + "="*80)
    print("TOP 10 STRATEGIES")
    print("="*80)

    sorted_results = sorted(all_results, key=lambda x: x['roi'], reverse=True)[:10]

    for i, result in enumerate(sorted_results, 1):
        print(f"\n#{i} - ROI: {result['roi']:.2f}%")
        print(f"    HIGH: {result['high_pct']*100:.1f}% | MEDIUM: {result['medium_pct']*100:.1f}% | LOW: {result['low_pct']*100:.1f}%")

    return best_params, all_results

def main():
    print("Loading historical performance data...")

    try:
        # Load OOS predictions
        oos_2024 = pd.read_csv('results/oos_predictions.csv')
        oos_2025 = pd.read_csv('results/predictions_2025_26.csv')
        historical = pd.concat([oos_2024, oos_2025], ignore_index=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure you have run the OOS tests first:")
        print("  python test_oos_performance.py")
        print("  python test_2025_26_season.py")
        return

    # Calculate confidence levels for each game
    print("Calculating confidence levels...")
    historical['confidence'] = historical.apply(get_confidence_level, axis=1)

    # Calculate correct predictions
    historical['correct'] = (historical['ensemble_pred'] > 0) == (historical['actual_margin'] > 0)

    # Run grid search
    best_params, all_results = grid_search_optimal_bets(historical, resolution=0.005)

    # Save results
    print("\nSaving results...")

    # Save optimal parameters
    with open('results/optimal_bet_sizes.json', 'w') as f:
        json.dump({
            'high_confidence_pct': best_params['high_pct'],
            'medium_confidence_pct': best_params['medium_pct'],
            'low_confidence_pct': best_params['low_pct'],
            'expected_roi': best_params['roi'],
            'historical_win_rate': best_params['win_rate']
        }, f, indent=2)

    print("Optimal bet sizes saved to results/optimal_bet_sizes.json")

    # Save all results for analysis
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/bet_sizing_grid_search.csv', index=False)
    print("Full grid search results saved to results/bet_sizing_grid_search.csv")

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
