"""
Optimize betting strategy with DIVERSIFIED allocation
Forces bets on all three confidence levels proportional to their edge
"""
import pandas as pd
import numpy as np
import json
from itertools import product

def get_confidence(row):
    """Determine confidence level based on model agreement"""
    preds = [row['elo_pred'], row['nn_pred'], row['xgb_pred']]
    spread = max(preds) - min(preds)
    avg_margin = abs(row['ensemble_pred'])

    if spread < 3 and avg_margin > 6:
        return "HIGH"
    elif spread < 6:
        return "MEDIUM"
    else:
        return "LOW"

def simulate_betting_strategy(historical_df, high_pct, medium_pct, low_pct, initial_bankroll=10000):
    """
    Simulate betting with FIXED percentages of initial bankroll
    """
    total_wagered = 0
    total_profit = 0
    num_bets = 0
    num_wins = 0

    results_by_confidence = {
        'HIGH': {'bets': 0, 'wins': 0, 'profit': 0, 'wagered': 0},
        'MEDIUM': {'bets': 0, 'wins': 0, 'profit': 0, 'wagered': 0},
        'LOW': {'bets': 0, 'wins': 0, 'profit': 0, 'wagered': 0}
    }

    for idx, row in historical_df.iterrows():
        confidence = row['confidence']

        if confidence == "HIGH":
            bet_pct = high_pct
        elif confidence == "MEDIUM":
            bet_pct = medium_pct
        else:
            bet_pct = low_pct

        if bet_pct == 0:
            continue

        bet_amount = initial_bankroll * bet_pct
        total_wagered += bet_amount
        num_bets += 1

        results_by_confidence[confidence]['bets'] += 1
        results_by_confidence[confidence]['wagered'] += bet_amount

        # Check if prediction was correct
        actual_correct = (row['ensemble_pred'] > 0) == (row['actual_margin'] > 0)

        if actual_correct:
            profit = bet_amount * (100/110)  # -110 odds
            total_profit += profit
            num_wins += 1
            results_by_confidence[confidence]['wins'] += 1
            results_by_confidence[confidence]['profit'] += profit
        else:
            total_profit -= bet_amount
            results_by_confidence[confidence]['profit'] -= bet_amount

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
        'by_confidence': results_by_confidence
    }

def grid_search_diversified(df, resolution=0.01):
    """
    Grid search with constraint: ALL three tiers must be > 0
    """
    # Use smaller range since we're betting on all three
    bet_range = np.arange(0.01, 0.15, resolution)  # 1% to 15%

    combinations = list(product(bet_range, bet_range, bet_range))

    print(f"Testing {len(combinations):,} combinations (all tiers > 0)...")

    results = []
    for high, medium, low in combinations:
        # Constraint: total allocation shouldn't exceed 30% of bankroll per game
        max_single_game = max(high, medium, low)
        if max_single_game > 0.15:  # No single tier > 15%
            continue

        result = simulate_betting_strategy(df, high, medium, low)
        results.append(result)

    # Find best by ROI
    best = max(results, key=lambda x: x['roi'])

    return best, results

if __name__ == "__main__":
    print("="*80)
    print("DIVERSIFIED BETTING STRATEGY OPTIMIZATION")
    print("Constraint: Bet on ALL three confidence levels")
    print("="*80)

    # Load data
    print("\nLoading data...")
    oos_2024 = pd.read_csv('results/predictions_fixed.csv')
    oos_2025 = pd.read_csv('results/predictions_2025_26.csv')

    # Add confidence levels
    oos_2024['confidence'] = oos_2024.apply(get_confidence, axis=1)
    oos_2025['confidence'] = oos_2025.apply(get_confidence, axis=1)

    print(f"Training set: {len(oos_2024)} games (2024-25)")
    print(f"Test set: {len(oos_2025)} games (2025-26)")

    # STEP 1: Optimize on training set
    print("\n" + "="*80)
    print("STEP 1: OPTIMIZE ON TRAINING SET (2024-25)")
    print("="*80)

    best_train, all_results = grid_search_diversified(oos_2024, resolution=0.01)

    print(f"\nOptimal Diversified Strategy:")
    print(f"  HIGH confidence: {best_train['high_pct']*100:.1f}%")
    print(f"  MEDIUM confidence: {best_train['medium_pct']*100:.1f}%")
    print(f"  LOW confidence: {best_train['low_pct']*100:.1f}%")
    print(f"\nTraining Performance:")
    print(f"  ROI: {best_train['roi']:.2f}%")
    print(f"  Win Rate: {best_train['win_rate']:.1%}")
    print(f"  Total Profit: ${best_train['total_profit']:,.2f}")
    print(f"  Total Bets: {best_train['num_bets']}")

    print(f"\nBreakdown by Confidence:")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        stats = best_train['by_confidence'][conf]
        if stats['bets'] > 0:
            conf_roi = (stats['profit'] / stats['wagered']) * 100
            conf_wr = stats['wins'] / stats['bets']
            print(f"  {conf:8s}: {stats['bets']:3d} bets, {conf_wr:.1%} win rate, {conf_roi:+.1f}% ROI")

    # STEP 2: Test on out-of-sample data
    print("\n" + "="*80)
    print("STEP 2: TEST ON OUT-OF-SAMPLE DATA (2025-26)")
    print("="*80)

    test_results = simulate_betting_strategy(
        oos_2025,
        best_train['high_pct'],
        best_train['medium_pct'],
        best_train['low_pct']
    )

    print(f"\nOut-of-Sample Performance:")
    print(f"  ROI: {test_results['roi']:.2f}%")
    print(f"  Win Rate: {test_results['win_rate']:.1%}")
    print(f"  Total Profit: ${test_results['total_profit']:,.2f}")
    print(f"  Total Bets: {test_results['num_bets']}")

    print(f"\nBreakdown by Confidence:")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        stats = test_results['by_confidence'][conf]
        if stats['bets'] > 0:
            conf_roi = (stats['profit'] / stats['wagered']) * 100 if stats['wagered'] > 0 else 0
            conf_wr = stats['wins'] / stats['bets'] if stats['bets'] > 0 else 0
            print(f"  {conf:8s}: {stats['bets']:3d} bets, {conf_wr:.1%} win rate, {conf_roi:+.1f}% ROI")

    # STEP 3: Calculate degradation
    print("\n" + "="*80)
    print("STEP 3: PERFORMANCE DEGRADATION")
    print("="*80)

    roi_degradation = test_results['roi'] - best_train['roi']
    wr_degradation = test_results['win_rate'] - best_train['win_rate']

    print(f"\nDegradation (Train -> Test):")
    print(f"  ROI Change: {roi_degradation:+.2f}% ({best_train['roi']:.2f}% -> {test_results['roi']:.2f}%)")
    print(f"  Win Rate Change: {wr_degradation:+.2%} ({best_train['win_rate']:.1%} -> {test_results['win_rate']:.1%})")

    if abs(roi_degradation) < 5:
        status = "[EXCELLENT]"
    elif abs(roi_degradation) < 10:
        status = "[GOOD]"
    elif abs(roi_degradation) < 15:
        status = "[MODERATE]"
    else:
        status = "[WARNING]"

    print(f"\nValidation Status: {status} Degradation of {abs(roi_degradation):.2f}%")

    # STEP 4: Compare to concentrated strategy
    print("\n" + "="*80)
    print("STEP 4: COMPARISON TO CONCENTRATED STRATEGY")
    print("="*80)

    # Load concentrated strategy results
    with open('results/optimal_bet_sizes_walkforward.json', 'r') as f:
        concentrated = json.load(f)

    print("\nConcentrated Strategy (LOW only):")
    print(f"  Allocation: HIGH=0%, MEDIUM=0%, LOW=8.5%")
    print(f"  Test ROI: {concentrated['out_of_sample_performance']['roi']:.2f}%")
    print(f"  Test Profit: ${concentrated['out_of_sample_performance']['total_profit']:,.2f}")

    print("\nDiversified Strategy (All three):")
    print(f"  Allocation: HIGH={best_train['high_pct']*100:.1f}%, MEDIUM={best_train['medium_pct']*100:.1f}%, LOW={best_train['low_pct']*100:.1f}%")
    print(f"  Test ROI: {test_results['roi']:.2f}%")
    print(f"  Test Profit: ${test_results['total_profit']:,.2f}")

    roi_diff = test_results['roi'] - concentrated['out_of_sample_performance']['roi']
    profit_diff = test_results['total_profit'] - concentrated['out_of_sample_performance']['total_profit']

    print(f"\nDifference:")
    print(f"  ROI: {roi_diff:+.2f}%")
    print(f"  Total Profit: ${profit_diff:+,.2f}")

    print("\n" + "="*80)
    print("TRADE-OFFS:")
    print("="*80)
    print("Concentrated (LOW only):")
    print("  + Higher ROI (more efficient)")
    print("  - Fewer betting opportunities")
    print("  - Lower total profit")
    print("\nDiversified (All tiers):")
    print("  + More betting opportunities")
    print("  + Potentially higher total profit")
    print("  - Slightly lower ROI")
    print("="*80)

    # Save results
    results_summary = {
        'diversified_strategy': {
            'high_confidence_pct': best_train['high_pct'],
            'medium_confidence_pct': best_train['medium_pct'],
            'low_confidence_pct': best_train['low_pct']
        },
        'in_sample_performance': {
            'dataset': '2024-25 season',
            'num_games': len(oos_2024),
            'roi': best_train['roi'],
            'win_rate': best_train['win_rate'],
            'total_profit': best_train['total_profit'],
            'num_bets': best_train['num_bets']
        },
        'out_of_sample_performance': {
            'dataset': '2025-26 season',
            'num_games': len(oos_2025),
            'roi': test_results['roi'],
            'win_rate': test_results['win_rate'],
            'total_profit': test_results['total_profit'],
            'num_bets': test_results['num_bets'],
            'by_confidence': test_results['by_confidence']
        },
        'degradation': {
            'roi_change': roi_degradation,
            'win_rate_change': wr_degradation
        },
        'comparison_to_concentrated': {
            'concentrated_roi': concentrated['out_of_sample_performance']['roi'],
            'diversified_roi': test_results['roi'],
            'roi_difference': roi_diff,
            'concentrated_profit': concentrated['out_of_sample_performance']['total_profit'],
            'diversified_profit': test_results['total_profit'],
            'profit_difference': profit_diff
        },
        'validation_method': 'walk_forward_diversified'
    }

    with open('results/optimal_bet_sizes_diversified.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\nResults saved to: results/optimal_bet_sizes_diversified.json")
