"""
Walk-Forward Bet Size Optimization
Properly validates betting strategy on truly out-of-sample data

Methodology:
1. Optimize bet sizes on 2024-25 season (1,225 games)
2. Validate optimized strategy on 2025-26 season (231 games)
3. Compare to in-sample optimization to detect overfitting
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
    """
    total_wagered = 0
    total_profit = 0
    num_bets = 0
    num_wins = 0

    # Track per-confidence performance
    confidence_stats = {
        'HIGH': {'bets': 0, 'wins': 0, 'wagered': 0, 'profit': 0},
        'MEDIUM': {'bets': 0, 'wins': 0, 'wagered': 0, 'profit': 0},
        'LOW': {'bets': 0, 'wins': 0, 'wagered': 0, 'profit': 0}
    }

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

        confidence_stats[confidence]['bets'] += 1
        confidence_stats[confidence]['wagered'] += bet_amount

        # Check if prediction was correct
        actual_correct = (row['ensemble_pred'] > 0) == (row['actual_margin'] > 0)

        if actual_correct:
            # Win: profit at -110 odds
            profit = bet_amount * (100/110)
            total_profit += profit
            num_wins += 1
            confidence_stats[confidence]['wins'] += 1
            confidence_stats[confidence]['profit'] += profit
        else:
            # Loss: lose the bet
            total_profit -= bet_amount
            confidence_stats[confidence]['profit'] -= bet_amount

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
        'low_pct': low_pct,
        'confidence_stats': confidence_stats
    }

def grid_search_optimal_bets(train_df, resolution=0.005):
    """
    Grid search to find optimal bet percentages on TRAINING data only
    """
    print(f"\nOptimizing on {len(train_df)} training games...")

    # Confidence distribution
    conf_counts = train_df['confidence'].value_counts()
    print(f"\nTraining Set Confidence Distribution:")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        if conf in conf_counts.index:
            count = conf_counts[conf]
            pct = count / len(train_df) * 100
            win_rate = sum((train_df[train_df['confidence']==conf]['correct'])) / count * 100
            print(f"  {conf}: {count} games ({pct:.1f}%) - Win Rate: {win_rate:.1f}%")

    # Define search ranges
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
                result = simulate_betting_strategy(
                    train_df, high_pct, medium_pct, low_pct
                )

                all_results.append(result)

                if result['roi'] > best_roi:
                    best_roi = result['roi']
                    best_params = result

                tested += 1

    print(f"Tested {tested:,} combinations")

    return best_params, all_results

def main():
    print("\n" + "="*80)
    print("WALK-FORWARD BET SIZE OPTIMIZATION")
    print("="*80)

    print("\nMethodology:")
    print("  1. TRAIN: Optimize bet sizes on 2024-25 season")
    print("  2. TEST: Validate on 2025-26 season (out-of-sample)")
    print("  3. COMPARE: In-sample vs out-of-sample performance")

    # Load data
    print("\nLoading data...")
    try:
        oos_2024 = pd.read_csv('results/oos_predictions.csv')
        oos_2025 = pd.read_csv('results/predictions_2025_26.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure you have run:")
        print("  python test_oos_performance.py")
        print("  python test_2025_26_season.py")
        return

    # Add confidence levels
    print("Calculating confidence levels...")
    oos_2024['confidence'] = oos_2024.apply(get_confidence_level, axis=1)
    oos_2025['confidence'] = oos_2025.apply(get_confidence_level, axis=1)

    # Calculate correct predictions
    oos_2024['correct'] = (oos_2024['ensemble_pred'] > 0) == (oos_2024['actual_margin'] > 0)
    oos_2025['correct'] = (oos_2025['ensemble_pred'] > 0) == (oos_2025['actual_margin'] > 0)

    print(f"\nData Loaded:")
    print(f"  Training (2024-25): {len(oos_2024)} games")
    print(f"  Testing (2025-26): {len(oos_2025)} games")

    # STEP 1: Optimize on training data (2024-25)
    print("\n" + "="*80)
    print("STEP 1: OPTIMIZATION ON 2024-25 SEASON (TRAINING)")
    print("="*80)

    best_params_train, all_results_train = grid_search_optimal_bets(oos_2024, resolution=0.005)

    print("\n" + "="*80)
    print("OPTIMAL BET SIZING (from 2024-25 training data)")
    print("="*80)
    print(f"\nHIGH Confidence: {best_params_train['high_pct']*100:.2f}% of bankroll")
    print(f"MEDIUM Confidence: {best_params_train['medium_pct']*100:.2f}% of bankroll")
    print(f"LOW Confidence: {best_params_train['low_pct']*100:.2f}% of bankroll")
    print(f"\nIn-Sample Performance (2024-25):")
    print(f"  ROI: {best_params_train['roi']:.2f}%")
    print(f"  Win Rate: {best_params_train['win_rate']:.2%}")
    print(f"  Total Profit: ${best_params_train['total_profit']:,.2f}")
    print(f"  Total Wagered: ${best_params_train['total_wagered']:,.2f}")

    # Show performance by confidence level (training)
    print(f"\nPer-Confidence Performance (2024-25 Training):")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        stats = best_params_train['confidence_stats'][conf]
        if stats['bets'] > 0:
            conf_win_rate = stats['wins'] / stats['bets']
            conf_roi = (stats['profit'] / stats['wagered'] * 100) if stats['wagered'] > 0 else 0
            print(f"  {conf}: {stats['bets']} bets, {conf_win_rate:.1%} win rate, {conf_roi:+.1f}% ROI")

    # STEP 2: Test on out-of-sample data (2025-26)
    print("\n" + "="*80)
    print("STEP 2: OUT-OF-SAMPLE VALIDATION ON 2025-26 SEASON")
    print("="*80)

    test_results = simulate_betting_strategy(
        oos_2025,
        best_params_train['high_pct'],
        best_params_train['medium_pct'],
        best_params_train['low_pct']
    )

    print("\nOut-of-Sample Performance (2025-26):")
    print(f"  ROI: {test_results['roi']:.2f}%")
    print(f"  Win Rate: {test_results['win_rate']:.2%}")
    print(f"  Total Profit: ${test_results['total_profit']:,.2f}")
    print(f"  Total Wagered: ${test_results['total_wagered']:,.2f}")
    print(f"  Total Bets: {test_results['num_bets']} ({test_results['num_wins']} wins)")

    # Show performance by confidence level (test)
    print(f"\nPer-Confidence Performance (2025-26 Test):")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        stats = test_results['confidence_stats'][conf]
        if stats['bets'] > 0:
            conf_win_rate = stats['wins'] / stats['bets']
            conf_roi = (stats['profit'] / stats['wagered'] * 100) if stats['wagered'] > 0 else 0
            print(f"  {conf}: {stats['bets']} bets, {conf_win_rate:.1%} win rate, {conf_roi:+.1f}% ROI")

    # STEP 3: Compare in-sample vs out-of-sample
    print("\n" + "="*80)
    print("STEP 3: IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("="*80)

    roi_degradation = test_results['roi'] - best_params_train['roi']
    win_rate_degradation = test_results['win_rate'] - best_params_train['win_rate']

    print(f"\nPerformance Degradation:")
    print(f"  ROI Change: {roi_degradation:+.2f}% ({best_params_train['roi']:.2f}% -> {test_results['roi']:.2f}%)")
    print(f"  Win Rate Change: {win_rate_degradation:+.2%} ({best_params_train['win_rate']:.2%} -> {test_results['win_rate']:.2%})")

    if abs(roi_degradation) < 5:
        print(f"\n[GOOD] Performance is stable (ROI degradation < 5%)")
        print("   The optimized strategy generalizes well to new data.")
    elif abs(roi_degradation) < 15:
        print(f"\n[MODERATE] Some performance degradation (ROI degradation {abs(roi_degradation):.1f}%)")
        print("   Strategy may be slightly overfit but still useful.")
    else:
        print(f"\n[WARNING] Significant performance degradation (ROI degradation {abs(roi_degradation):.1f}%)")
        print("   Strategy appears overfit to training data.")

    # STEP 4: Compare to baseline (equal betting)
    print("\n" + "="*80)
    print("STEP 4: COMPARISON TO BASELINE STRATEGY")
    print("="*80)

    # Equal betting (2% on all games)
    baseline_train = simulate_betting_strategy(oos_2024, 0.02, 0.02, 0.02)
    baseline_test = simulate_betting_strategy(oos_2025, 0.02, 0.02, 0.02)

    print("\nBaseline Strategy (2% on all games):")
    print(f"  Training ROI: {baseline_train['roi']:.2f}%")
    print(f"  Test ROI: {baseline_test['roi']:.2f}%")

    print("\nOptimized Strategy:")
    print(f"  Training ROI: {best_params_train['roi']:.2f}%")
    print(f"  Test ROI: {test_results['roi']:.2f}%")

    improvement_train = best_params_train['roi'] - baseline_train['roi']
    improvement_test = test_results['roi'] - baseline_test['roi']

    print(f"\nImprovement vs Baseline:")
    print(f"  Training: {improvement_train:+.2f}% ({baseline_train['roi']:.2f}% -> {best_params_train['roi']:.2f}%)")
    print(f"  Test: {improvement_test:+.2f}% ({baseline_test['roi']:.2f}% -> {test_results['roi']:.2f}%)")

    if improvement_test > 0:
        print(f"\n[SUCCESS] Optimized strategy beats baseline on out-of-sample data by {improvement_test:.2f}%")
    else:
        print(f"\n[FAIL] Baseline strategy performs better on out-of-sample data by {abs(improvement_test):.2f}%")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_summary = {
        'optimal_strategy': {
            'high_confidence_pct': best_params_train['high_pct'],
            'medium_confidence_pct': best_params_train['medium_pct'],
            'low_confidence_pct': best_params_train['low_pct']
        },
        'in_sample_performance': {
            'dataset': '2024-25 season',
            'num_games': len(oos_2024),
            'roi': best_params_train['roi'],
            'win_rate': best_params_train['win_rate'],
            'total_profit': best_params_train['total_profit']
        },
        'out_of_sample_performance': {
            'dataset': '2025-26 season',
            'num_games': len(oos_2025),
            'roi': test_results['roi'],
            'win_rate': test_results['win_rate'],
            'total_profit': test_results['total_profit']
        },
        'degradation': {
            'roi_change': roi_degradation,
            'win_rate_change': win_rate_degradation
        },
        'baseline_comparison': {
            'baseline_test_roi': baseline_test['roi'],
            'optimized_test_roi': test_results['roi'],
            'improvement': improvement_test
        },
        'validation_method': 'walk_forward',
        'training_period': '2024-25',
        'test_period': '2025-26'
    }

    with open('results/optimal_bet_sizes_walkforward.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("Results saved to results/optimal_bet_sizes_walkforward.json")

    # Also save detailed grid search results
    results_df_train = pd.DataFrame(all_results_train)
    results_df_train.to_csv('results/bet_sizing_grid_search_walkforward.csv', index=False)
    print("Grid search results saved to results/bet_sizing_grid_search_walkforward.csv")

    print("\n" + "="*80)
    print("WALK-FORWARD OPTIMIZATION COMPLETE")
    print("="*80)

    print("\n[SUMMARY]")
    print(f"  Optimal Strategy: HIGH {best_params_train['high_pct']*100:.1f}% | MEDIUM {best_params_train['medium_pct']*100:.1f}% | LOW {best_params_train['low_pct']*100:.1f}%")
    print(f"  Out-of-Sample ROI: {test_results['roi']:.2f}%")
    print(f"  Beats Baseline: {'YES' if improvement_test > 0 else 'NO'} ({improvement_test:+.2f}%)")
    print(f"  Overfitting: {'Minimal' if abs(roi_degradation) < 5 else 'Moderate' if abs(roi_degradation) < 15 else 'Significant'}")

if __name__ == "__main__":
    main()
