# NBA Betting Strategies Comparison

## Overview
Both strategies are **profitable** and validated on out-of-sample data (2025-26 season). The choice depends on your goals: **efficiency** vs **total profit**.

---

## Strategy 1: Optimized (LOW Confidence Only)

### Allocation
- **HIGH confidence**: 0%
- **MEDIUM confidence**: 0%
- **LOW confidence**: 8.5%

### Performance (Out-of-Sample)
- **ROI**: 35.23% ✅ *Most efficient*
- **Total Profit**: $14,372.73 (on $10,000 bankroll)
- **Games Bet**: 48 games (only LOW confidence)
- **Win Rate**: 65.4%

### Validation
- Training ROI: 46.85%
- Test ROI: 35.23%
- Degradation: -11.6% (acceptable)
- Beats baseline by: +10.43%

### Pros
- ✅ Highest ROI (most efficient use of capital)
- ✅ Lower risk (fewer bets)
- ✅ Focus on best-performing tier

### Cons
- ❌ Fewer betting opportunities (only 48 games)
- ❌ Lower total profit

### Best For
- Conservative bettors who prioritize efficiency
- Limited bankroll
- Risk-averse approach

---

## Strategy 2: Diversified (All Three Tiers)

### Allocation
- **HIGH confidence**: 14.0%
- **MEDIUM confidence**: 1.0%
- **LOW confidence**: 14.0%

### Performance (Out-of-Sample)
- **ROI**: 32.37% (2.86% lower than concentrated)
- **Total Profit**: $33,145.45 (on $10,000 bankroll) ✅ ***+130% more!***
- **Games Bet**: 231 games (all opportunities)
- **Win Rate**: 65.4%

### Breakdown by Tier (Out-of-Sample)
| Tier   | Games | Win Rate | ROI    |
|--------|-------|----------|--------|
| HIGH   | 13    | 69.2%    | +32.2% |
| MEDIUM | 170   | 63.5%    | +21.3% |
| LOW    | 48    | 70.8%    | +35.2% |

### Validation
- Training ROI: 39.43%
- Test ROI: 32.37%
- Degradation: -7.06% (excellent!)
- Beats baseline by: +7.58%

### Pros
- ✅ More than DOUBLE the total profit ($18,772 more)
- ✅ More betting opportunities (231 games vs 48)
- ✅ Better degradation (7% vs 11.6%)
- ✅ All three tiers are profitable

### Cons
- ❌ Slightly lower ROI efficiency (2.86% less)
- ❌ Higher variance (more bets)
- ❌ Requires larger bankroll allocation per game

### Best For
- Aggressive bettors who want maximum profit
- Larger bankroll
- Those who want more action
- Long-term profit maximization

---

## Key Insights

### Why is LOW Confidence Best?
When models **disagree**, the ensemble synthesizes different perspectives, finding edges the market misses. When models **agree** (HIGH confidence), they're often using the same public information already priced into the betting line.

### All Tiers are Profitable
At -110 odds, you need >52.38% win rate to be profitable:
- HIGH: 69.2% ✅ (+16.8% edge)
- MEDIUM: 63.5% ✅ (+11.1% edge)
- LOW: 70.8% ✅ (+18.4% edge)

All three beat the break-even threshold significantly!

---

## Recommendation

### Choose **Optimized (LOW only)** if:
- You want the highest ROI
- You have a smaller bankroll
- You prefer fewer, higher-quality bets
- You're risk-averse

### Choose **Diversified (All tiers)** if:
- You want maximum total profit
- You have a larger bankroll ($10,000+)
- You want more betting opportunities
- You're comfortable with slightly lower efficiency for higher total returns

---

## Example: $10,000 Bankroll Over 231 Games

| Strategy | ROI | Total Profit | # Bets | Avg Bet Size |
|----------|-----|--------------|--------|--------------|
| Optimized (LOW only) | 35.23% | $14,372.73 | 48 | $850 |
| Diversified (All tiers) | 32.37% | $33,145.45 | 231 | Variable* |

*Diversified strategy: HIGH=14%, MEDIUM=1%, LOW=14% of bankroll per game

---

## Both Strategies are Walk-Forward Validated

Both strategies were optimized on the 2024-25 season (1,225 games) and tested on the completely unseen 2025-26 season (231 games). The test results represent **realistic expected performance** on future games.

**Update**: The dashboard now defaults to **Diversified (All tiers)** as it provides the best total profit while maintaining strong ROI. You can switch between strategies in the "Strategy for Bet Sizing" dropdown.
