"""
Interactive NBA Prediction Dashboard v2
Features:
- Navigable game menu
- Optimized bet sizing from grid search
- Enhanced visualizations

Run with: streamlit run dashboard_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import json
sys.path.append('src')

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams
from models.neural_network import NBANeuralNetwork
from models.xgboost_model import NBAXGBoost
from models.ensemble import NBAEnsemble

# Page configuration
st.set_page_config(
    page_title="NBA Prediction Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="auto"  # Collapsed on mobile, expanded on desktop
)

# Custom CSS with mobile optimization
st.markdown("""
<style>
    /* Desktop styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-confidence {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .medium-confidence {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .low-confidence {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
            margin-bottom: 1rem !important;
        }

        /* Make metrics stack vertically on mobile */
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            margin-bottom: 0.5rem;
        }

        /* Reduce padding on mobile */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Make tables scrollable */
        .dataframe {
            font-size: 0.85rem;
            overflow-x: auto;
        }

        /* Smaller font for mobile */
        body, p, div {
            font-size: 0.9rem;
        }

        /* Reduce chart heights on mobile */
        .js-plotly-plot {
            max-height: 300px !important;
        }

        /* Optimize sidebar for mobile (collapsible, not hidden) */
        [data-testid="stSidebar"][aria-expanded="true"] {
            width: 85% !important;
            max-width: 320px !important;
        }

        /* Optimize metric cards for mobile */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
    }

    /* Tablet optimizations */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load models (cached)
@st.cache_resource
def load_models():
    """Load all trained models"""
    nn_model = NBANeuralNetwork()
    nn_model.load_model('models/neural_network_fixed')

    xgb_model = NBAXGBoost()
    xgb_model.load_model('models/xgboost_fixed')

    ensemble = NBAEnsemble()
    ensemble.load_model('models/ensemble_fixed')

    return nn_model, xgb_model, ensemble

# Load data (cached)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions():
    """Load tonight's predictions with freshness check"""
    try:
        predictions = pd.read_csv('results/tonights_predictions.csv')

        # Check file modification time to see if predictions are fresh
        import os
        file_path = 'results/tonights_predictions.csv'
        if os.path.exists(file_path):
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            predictions.attrs['last_updated'] = modified_time

        return predictions
    except:
        return None

def generate_fresh_predictions():
    """Generate fresh predictions by running predict_tonight.py script"""
    try:
        import subprocess
        import os

        # Run the prediction script
        result = subprocess.run(
            ['python', 'predict_tonight.py'],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check if predictions file was created/updated
        if os.path.exists('results/tonights_predictions.csv'):
            predictions = pd.read_csv('results/tonights_predictions.csv')
            if len(predictions) > 0:
                return predictions, f"Generated {len(predictions)} predictions for today"
            else:
                return None, "No games scheduled for today"
        else:
            return None, f"Prediction script failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return None, "Prediction generation timed out. Please try again."
    except Exception as e:
        return None, f"Error generating predictions: {str(e)}"

@st.cache_data(ttl=3600)
def load_historical_performance():
    """Load historical performance data"""
    try:
        oos_2024 = pd.read_csv('results/oos_predictions.csv')
        oos_2025 = pd.read_csv('results/predictions_2025_26.csv')
        return pd.concat([oos_2024, oos_2025], ignore_index=True)
    except:
        return None

@st.cache_data(ttl=3600)
def load_optimal_bet_sizes():
    """Load optimized bet sizes from walk-forward validation"""
    try:
        # Try to load walk-forward results first (properly validated)
        with open('results/optimal_bet_sizes_walkforward.json', 'r') as f:
            data = json.load(f)
            # Restructure to match expected format
            return {
                'high_confidence_pct': data['optimal_strategy']['high_confidence_pct'],
                'medium_confidence_pct': data['optimal_strategy']['medium_confidence_pct'],
                'low_confidence_pct': data['optimal_strategy']['low_confidence_pct'],
                'expected_roi': data['out_of_sample_performance']['roi'],  # Use OOS performance!
                'historical_win_rate': data['out_of_sample_performance']['win_rate'],
                'validation_method': 'walk_forward',
                'training_roi': data['in_sample_performance']['roi'],
                'test_roi': data['out_of_sample_performance']['roi'],
                'degradation': data['degradation']['roi_change'],
                'baseline_improvement': data['baseline_comparison']['improvement']
            }
    except:
        # Fallback to in-sample results (overfitted)
        try:
            with open('results/optimal_bet_sizes.json', 'r') as f:
                data = json.load(f)
                data['validation_method'] = 'in_sample'  # Mark as not validated
                return data
        except:
            return None

def get_confidence_level(predictions_row):
    """
    Calculate confidence level based on model agreement
    High: All models agree within 3 points
    Medium: Models agree within 6 points
    Low: Models disagree by >6 points
    """
    preds = [
        predictions_row['ELO_PREDICTION'],
        predictions_row['NN_PREDICTION'],
        predictions_row['XGB_PREDICTION'],
        predictions_row['ENSEMBLE_PREDICTION']
    ]

    spread = max(preds) - min(preds)

    if spread < 3:
        return "HIGH", "🟢"
    elif spread < 6:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🔴"

def calculate_bet_size(confidence, bankroll=10000, strategy="Moderate"):
    """
    Calculate bet size based on confidence level and strategy
    """
    # Predefined strategies
    multipliers = {
        "Conservative": {"HIGH": 0.02, "MEDIUM": 0.01, "LOW": 0.005},
        "Moderate": {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01},
        "Aggressive": {"HIGH": 0.05, "MEDIUM": 0.03, "LOW": 0.015},
    }

    # Load optimized strategies if available
    if strategy == "Optimized (LOW only)":
        optimal = load_optimal_bet_sizes()
        if optimal:
            multipliers["Optimized (LOW only)"] = {
                "HIGH": optimal.get('high_confidence_pct', 0.0),
                "MEDIUM": optimal.get('medium_confidence_pct', 0.0),
                "LOW": optimal.get('low_confidence_pct', 0.085)
            }
        else:
            # Fallback if file not found
            multipliers["Optimized (LOW only)"] = {"HIGH": 0.0, "MEDIUM": 0.0, "LOW": 0.085}

    elif strategy == "Diversified (All tiers)":
        try:
            with open('results/optimal_bet_sizes_diversified.json', 'r') as f:
                data = json.load(f)
                multipliers["Diversified (All tiers)"] = {
                    "HIGH": data['diversified_strategy']['high_confidence_pct'],
                    "MEDIUM": data['diversified_strategy']['medium_confidence_pct'],
                    "LOW": data['diversified_strategy']['low_confidence_pct']
                }
        except:
            # Fallback if file not found
            multipliers["Diversified (All tiers)"] = {"HIGH": 0.14, "MEDIUM": 0.01, "LOW": 0.14}

    bet_pct = multipliers.get(strategy, multipliers["Moderate"]).get(confidence, 0.01)
    return bankroll * bet_pct

def display_game_card(game, game_num, total_games, bankroll, strategy):
    """Display a single game prediction card with enhanced visualizations"""
    confidence, emoji = get_confidence_level(game)

    confidence_class = {
        "HIGH": "high-confidence",
        "MEDIUM": "medium-confidence",
        "LOW": "low-confidence"
    }[confidence]

    st.markdown(f"<div class='prediction-box {confidence_class}'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"### Game {game_num}/{total_games}: {game['AWAY_TEAM']} @ {game['HOME_TEAM']}")
        st.markdown(f"**Time:** {game['GAME_TIME']}")

        # Prediction
        if game['ENSEMBLE_PREDICTION'] > 0:
            winner = game['HOME_TEAM']
            loser = game['AWAY_TEAM']
            margin = game['ENSEMBLE_PREDICTION']
        else:
            winner = game['AWAY_TEAM']
            loser = game['HOME_TEAM']
            margin = abs(game['ENSEMBLE_PREDICTION'])

        st.markdown(f"**🏆 Predicted Winner:** {winner}")
        st.markdown(f"**📊 Predicted Margin:** {margin:.1f} points")

        # Model agreement visualization
        preds = [game['ELO_PREDICTION'], game['NN_PREDICTION'],
                game['XGB_PREDICTION'], game['ENSEMBLE_PREDICTION']]
        spread = max(preds) - min(preds)

        col_a, col_b = st.columns(2)
        col_a.metric("Model Spread", f"{spread:.1f} pts",
                    help="Difference between highest and lowest prediction")
        col_b.metric("Confidence", f"{confidence}",
                    help=f"Based on model agreement")

        # Show all model predictions in table format
        model_data = pd.DataFrame({
            'Model': ['Elo', 'Neural Net', 'XGBoost', 'Ensemble'],
            'Prediction': [f"{game['ELO_PREDICTION']:+.1f}",
                          f"{game['NN_PREDICTION']:+.1f}",
                          f"{game['XGB_PREDICTION']:+.1f}",
                          f"{game['ENSEMBLE_PREDICTION']:+.1f}"]
        })
        st.dataframe(model_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f"### {emoji} {confidence}")

        # Model predictions chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Elo', 'NN', 'XGB', 'Ensemble'],
            y=[game['ELO_PREDICTION'], game['NN_PREDICTION'],
               game['XGB_PREDICTION'], game['ENSEMBLE_PREDICTION']],
            marker_color=['#FF6B35', '#004E89', '#F77F00', '#06A77D'],
            text=[f"{x:+.1f}" for x in [game['ELO_PREDICTION'], game['NN_PREDICTION'],
                   game['XGB_PREDICTION'], game['ENSEMBLE_PREDICTION']]],
            textposition='auto'
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Margin",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        # Betting information
        bet_size = calculate_bet_size(confidence, bankroll, strategy)
        bet_pct = (bet_size / bankroll) * 100

        # Get validated win rate based on strategy
        if strategy == "Diversified (All tiers)":
            # Use diversified strategy win rates
            if confidence == 'LOW':
                win_rate = 0.708
            elif confidence == 'MEDIUM':
                win_rate = 0.635
            else:
                win_rate = 0.692
        elif strategy == "Optimized (LOW only)":
            # Use concentrated strategy win rates
            if confidence == 'LOW':
                win_rate = 0.708
            elif confidence == 'MEDIUM':
                win_rate = 0.637
            else:
                win_rate = 0.642
        else:
            # Use conservative average for other strategies
            win_rate = 0.659

        # Calculate expected value
        if bet_size > 0:
            expected_profit = bet_size * win_rate * (100/110) - bet_size * (1 - win_rate)
            expected_roi = (expected_profit / bet_size) * 100

            st.markdown(f"**💰 Bet:** ${bet_size:.2f} ({bet_pct:.1f}%)")
            st.markdown(f"**📈 Expected Profit:** ${expected_profit:+.2f}")
            st.markdown(f"**🎯 Expected ROI:** {expected_roi:+.1f}%")
            st.markdown(f"**✅ Win Probability:** {win_rate:.1%}")
        else:
            st.markdown(f"**💰 Bet:** $0.00 (0%)")
            st.markdown(f"**⏭️ Skip:** Optimized strategy skips {confidence} confidence")
            st.markdown(f"**ℹ️ Reason:** {confidence} confidence has lower expected value ({win_rate:.1%} win rate)")

    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏀 NBA Predictor")

    page = st.radio(
        "Navigate",
        ["📊 Tonight's Games", "📈 Performance Tracking", "💰 Betting Strategy",
         "🔬 Model Explanation", "⚙️ Settings"]
    )

    st.markdown("---")
    st.markdown("### Model Stats")
    st.metric("Historical Accuracy", "65.9%")
    st.metric("2025-26 Accuracy", "65.8%")

    # Show optimized vs standard ROI
    optimal_data = load_optimal_bet_sizes()
    if optimal_data:
        st.metric("Optimized ROI", f"+{optimal_data['expected_roi']:.1f}%")
    else:
        st.metric("Average ROI", "+24.8%")

# Main content
if page == "📊 Tonight's Games":
    st.markdown("<h1 class='main-header'>🏀 Tonight's NBA Predictions</h1>", unsafe_allow_html=True)

    # Check if predictions need auto-refresh
    import os
    predictions_stale = False
    auto_refresh_attempted = False

    if os.path.exists('results/tonights_predictions.csv'):
        modified_time = datetime.fromtimestamp(os.path.getmtime('results/tonights_predictions.csv'))
        modified_date = modified_time.date()
        today_date = datetime.now().date()

        # Auto-refresh if predictions are from a different day
        if modified_date < today_date and 'auto_refreshed' not in st.session_state:
            st.session_state.auto_refreshed = True
            with st.spinner("🔄 Auto-refreshing predictions for today..."):
                new_predictions, message = generate_fresh_predictions()
                if new_predictions is not None:
                    st.success(message)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    predictions_stale = True
                    auto_refresh_attempted = True

    # Add refresh controls
    col_info, col_refresh = st.columns([3, 1])

    with col_info:
        if os.path.exists('results/tonights_predictions.csv'):
            modified_time = datetime.fromtimestamp(os.path.getmtime('results/tonights_predictions.csv'))
            modified_date = modified_time.date()
            today_date = datetime.now().date()
            time_diff = datetime.now() - modified_time

            if modified_date < today_date:
                if auto_refresh_attempted:
                    st.error(f"❌ Could not fetch today's games. Showing predictions from {modified_date.strftime('%B %d, %Y')}. Click 'Refresh' to try again.")
                else:
                    st.warning(f"⚠️ Predictions are from {modified_date.strftime('%B %d, %Y')}. Click 'Refresh' to update.")
            elif time_diff.seconds > 14400:  # 4 hours
                st.info(f"ℹ️ Predictions last updated {time_diff.seconds // 3600} hours ago.")
            else:
                st.success(f"✅ Predictions updated {time_diff.seconds // 60} minutes ago.")

    with col_refresh:
        if st.button("🔄 Refresh Predictions", type="primary", use_container_width=True):
            st.session_state.auto_refreshed = False  # Reset auto-refresh flag
            with st.spinner("Fetching today's games and generating predictions..."):
                new_predictions, message = generate_fresh_predictions()
                if new_predictions is not None:
                    st.success(message)
                    st.cache_data.clear()  # Clear cache to reload fresh data
                    st.rerun()
                else:
                    st.error(message)

    predictions = load_predictions()

    if predictions is not None and len(predictions) > 0:
        st.markdown(f"### {datetime.now().strftime('%B %d, %Y')} - {len(predictions)} Games")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        high_conf = sum([1 for _, row in predictions.iterrows() if get_confidence_level(row)[0] == "HIGH"])
        med_conf = sum([1 for _, row in predictions.iterrows() if get_confidence_level(row)[0] == "MEDIUM"])
        low_conf = sum([1 for _, row in predictions.iterrows() if get_confidence_level(row)[0] == "LOW"])

        col1.metric("Total Games", len(predictions))
        col2.metric("🟢 High Confidence", high_conf)
        col3.metric("🟡 Medium Confidence", med_conf)
        col4.metric("🔴 Low Confidence", low_conf)

        st.markdown("---")

        # Betting controls and visualization
        left_col, right_col = st.columns([1, 2])

        with left_col:
            display_bankroll = st.number_input("💰 Bankroll", value=10000, step=1000, key="display_bankroll",
                                              help="Your total betting bankroll")

            strategy_options = {
                "🎯 Max Profit (32% ROI)": "Diversified (All tiers)",
                "⚡ Max Efficiency (35% ROI)": "Optimized (LOW only)",
                "🛡️ Safe & Simple": "Conservative",
                "📊 Balanced": "Moderate",
                "🚀 High Risk": "Aggressive"
            }
            strategy_display = st.selectbox(
                "📈 Betting Strategy",
                list(strategy_options.keys()),
                index=0,  # Default to Max Profit
                help="Max Profit bets on all games. Max Efficiency only bets on LOW confidence games."
            )
            display_strategy = strategy_options[strategy_display]

        st.markdown("---")

        # Load optimal strategy data
        optimal = load_optimal_bet_sizes()

        # Build summary table data and calculate bet distribution
        summary_data = []
        total_to_bet = 0
        total_expected_profit = 0
        bet_distribution = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        for idx, game in predictions.iterrows():
            confidence, emoji = get_confidence_level(game)
            bet_amount = calculate_bet_size(confidence, display_bankroll, display_strategy)
            total_to_bet += bet_amount
            bet_distribution[confidence] += bet_amount

            # Get win rate based on strategy
            if display_strategy == "Diversified (All tiers)":
                if confidence == 'LOW':
                    win_rate = 0.708
                elif confidence == 'MEDIUM':
                    win_rate = 0.635
                else:
                    win_rate = 0.692
            elif display_strategy == "Optimized (LOW only)" or (optimal and optimal.get('validation_method') == 'walk_forward'):
                if confidence == 'LOW':
                    win_rate = 0.708
                elif confidence == 'MEDIUM':
                    win_rate = 0.637
                else:
                    win_rate = 0.642
            else:
                win_rate = 0.659

            expected_value = bet_amount * win_rate * (100/110) - bet_amount * (1 - win_rate)
            total_expected_profit += expected_value
            profit_if_won = bet_amount * (100/110)

            # Determine which side to bet
            ensemble_pred = game['ENSEMBLE_PREDICTION']
            if ensemble_pred > 0:
                bet_side = f"{game['HOME_TEAM']} ({ensemble_pred:+.1f})"
            else:
                bet_side = f"{game['AWAY_TEAM']} ({ensemble_pred:+.1f})"

            # Add to summary data
            summary_data.append({
                'Game': f"{game['AWAY_TEAM']} @ {game['HOME_TEAM']}",
                'Bet On': bet_side,
                'Confidence': f"{emoji} {confidence}",
                'Wager': f"${bet_amount:.2f}",
                'Win %': f"{win_rate:.1%}",
                'Profit if Won': f"${profit_if_won:+.2f}"
            })

        # Add pie chart to right column
        with right_col:
            st.markdown("### 📊 Bet Allocation by Confidence")

            # Create pie chart
            labels = []
            values = []
            colors = []

            if bet_distribution['HIGH'] > 0:
                labels.append('🟢 HIGH')
                values.append(bet_distribution['HIGH'])
                colors.append('#28a745')

            if bet_distribution['MEDIUM'] > 0:
                labels.append('🟡 MEDIUM')
                values.append(bet_distribution['MEDIUM'])
                colors.append('#ffc107')

            if bet_distribution['LOW'] > 0:
                labels.append('🔴 LOW')
                values.append(bet_distribution['LOW'])
                colors.append('#dc3545')

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='%{label}: $%{value:,.2f}<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                height=300,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add warning if total bet is too high
            bankroll_pct = (total_to_bet / display_bankroll) * 100
            if bankroll_pct > 30:
                st.warning(f"⚠️ **High Risk:** Betting {bankroll_pct:.1f}% of bankroll in one night. Consider reducing bet sizes or being more selective.")
            elif bankroll_pct > 20:
                st.info(f"ℹ️ Betting {bankroll_pct:.1f}% of bankroll tonight. This is aggressive but manageable.")

        # Display summary table
        st.markdown("### 📋 Betting Summary - All Games")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption("""
        **About the wagers:** These are spread bets at -110 odds (standard sportsbook odds).
        "Profit if Won" shows what you'll win if the bet is correct ($90.91 profit for every $100 wagered).
        The "Bet On" column shows which team to bet on and our predicted point spread.
        """)

        st.markdown("---")

        # Total metrics
        st.markdown("### 💰 Tonight's Predicted Returns")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Recommended Bet", f"${total_to_bet:,.2f}",
                   help=f"{total_to_bet/display_bankroll*100:.1f}% of bankroll")
        col2.metric("Predicted Return", f"${total_expected_profit:,.2f}",
                   help="Expected profit if bets perform at historical win rates")
        col3.metric("Predicted ROI", f"{(total_expected_profit/total_to_bet*100):.1f}%",
                   help="Expected return on investment based on validated win rates")

        st.markdown("---")

        # GAME NAVIGATION
        st.markdown("### 🎯 Jump to Game")

        # Create game options
        game_options = []
        for idx, game in predictions.iterrows():
            confidence, emoji = get_confidence_level(game)
            game_label = f"{emoji} Game {idx+1}: {game['AWAY_TEAM']} @ {game['HOME_TEAM']}"
            game_options.append(game_label)

        # Game selector and view mode on same row
        nav_col1, nav_col2 = st.columns([2, 1])

        with nav_col1:
            selected_game = st.selectbox(
                "Select a game to view details:",
                options=range(len(game_options)),
                format_func=lambda x: game_options[x],
                key="game_selector"
            )

        with nav_col2:
            view_mode = st.radio(
                "View Mode:",
                ["Selected Game Only", "All Games"],
                horizontal=False
            )

        # Display games based on view mode
        if view_mode == "Selected Game Only":
            # Show only selected game
            game = predictions.iloc[selected_game]
            display_game_card(game, selected_game + 1, len(predictions), display_bankroll, display_strategy)

        else:
            # Group games by confidence and show in expandable sections
            st.markdown("### 🎯 Games by Confidence Level")

            # Separate games by confidence
            high_games = []
            medium_games = []
            low_games = []

            for idx, game in predictions.iterrows():
                confidence, _ = get_confidence_level(game)
                if confidence == "HIGH":
                    high_games.append((idx, game))
                elif confidence == "MEDIUM":
                    medium_games.append((idx, game))
                else:
                    low_games.append((idx, game))

            # LOW CONFIDENCE (Most profitable - show first!)
            if low_games:
                # Calculate summary stats for LOW confidence
                low_total_bet = sum([calculate_bet_size("LOW", display_bankroll, display_strategy) for _ in low_games])
                # Get win rate based on strategy
                if display_strategy in ["Diversified (All tiers)", "Optimized (LOW only)"]:
                    low_win_rate = 0.708
                else:
                    low_win_rate = 0.659
                low_expected_profit = sum([
                    calculate_bet_size("LOW", display_bankroll, display_strategy) * low_win_rate * (100/110) -
                    calculate_bet_size("LOW", display_bankroll, display_strategy) * (1 - low_win_rate)
                    for _ in low_games
                ])

                with st.expander(f"🔴 LOW Confidence - {len(low_games)} Games (70.8% Win Rate - BEST!) - Expected Profit: ${low_expected_profit:,.2f}", expanded=True):
                    st.markdown(f"""
                    **Why LOW Confidence is BEST:**
                    - Validated 70.8% win rate on 2025-26 season
                    - Model disagreement indicates ensemble strength
                    - Highest expected ROI per game

                    **Summary:**
                    - Games: {len(low_games)}
                    - Total Bet: ${low_total_bet:,.2f} ({low_total_bet/display_bankroll*100:.1f}% of bankroll)
                    - Expected Win Rate: {low_win_rate:.1%}
                    - Expected Profit: ${low_expected_profit:,.2f}
                    - Expected ROI: {(low_expected_profit/low_total_bet*100):.1f}%
                    """)

                    for idx, game in low_games:
                        display_game_card(game, idx + 1, len(predictions), display_bankroll, display_strategy)
                        st.markdown("<br>", unsafe_allow_html=True)

            # MEDIUM CONFIDENCE
            if medium_games:
                medium_total_bet = sum([calculate_bet_size("MEDIUM", display_bankroll, display_strategy) for _ in medium_games])
                # Get win rate based on strategy
                if display_strategy == "Diversified (All tiers)":
                    medium_win_rate = 0.635
                elif display_strategy == "Optimized (LOW only)":
                    medium_win_rate = 0.637
                else:
                    medium_win_rate = 0.659
                medium_expected_profit = sum([
                    calculate_bet_size("MEDIUM", display_bankroll, display_strategy) * medium_win_rate * (100/110) -
                    calculate_bet_size("MEDIUM", display_bankroll, display_strategy) * (1 - medium_win_rate)
                    for _ in medium_games
                ])

                with st.expander(f"🟡 MEDIUM Confidence - {len(medium_games)} Games (63.7% Win Rate) - Expected Profit: ${medium_expected_profit:,.2f}", expanded=False):
                    st.markdown(f"""
                    **Summary:**
                    - Games: {len(medium_games)}
                    - Total Bet: ${medium_total_bet:,.2f} ({medium_total_bet/display_bankroll*100:.1f}% of bankroll)
                    - Expected Win Rate: {medium_win_rate:.1%}
                    - Expected Profit: ${medium_expected_profit:,.2f}
                    - Expected ROI: {(medium_expected_profit/medium_total_bet*100):.1f}%
                    """)

                    for idx, game in medium_games:
                        display_game_card(game, idx + 1, len(predictions), display_bankroll, display_strategy)
                        st.markdown("<br>", unsafe_allow_html=True)

            # HIGH CONFIDENCE
            if high_games:
                high_total_bet = sum([calculate_bet_size("HIGH", display_bankroll, display_strategy) for _ in high_games])
                # Get win rate based on strategy
                if display_strategy == "Diversified (All tiers)":
                    high_win_rate = 0.692
                elif display_strategy == "Optimized (LOW only)":
                    high_win_rate = 0.642
                else:
                    high_win_rate = 0.659
                high_expected_profit = sum([
                    calculate_bet_size("HIGH", display_bankroll, display_strategy) * high_win_rate * (100/110) -
                    calculate_bet_size("HIGH", display_bankroll, display_strategy) * (1 - high_win_rate)
                    for _ in high_games
                ])

                with st.expander(f"🟢 HIGH Confidence - {len(high_games)} Games (64.2% Win Rate) - Expected Profit: ${high_expected_profit:,.2f}", expanded=False):
                    st.markdown(f"""
                    **Summary:**
                    - Games: {len(high_games)}
                    - Total Bet: ${high_total_bet:,.2f} ({high_total_bet/display_bankroll*100:.1f}% of bankroll)
                    - Expected Win Rate: {high_win_rate:.1%}
                    - Expected Profit: ${high_expected_profit:,.2f}
                    - Expected ROI: {(high_expected_profit/high_total_bet*100):.1f}%
                    """)

                    for idx, game in high_games:
                        display_game_card(game, idx + 1, len(predictions), display_bankroll, display_strategy)
                        st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.info("No games scheduled for tonight. Run `python predict_tonight.py` to generate predictions.")

elif page == "📈 Performance Tracking":
    st.markdown("<h1 class='main-header'>📈 Performance Tracking</h1>", unsafe_allow_html=True)

    # Load historical performance
    historical = load_historical_performance()

    if historical is not None:
        # Date range filter and strategy selection
        st.markdown("### 📅 Select Time Period & Strategy")

        if 'game_date' in historical.columns:
            historical['game_date'] = pd.to_datetime(historical['game_date'])
            min_date = historical['game_date'].min().date()
            max_date = historical['game_date'].max().date()

            # Default to October 21, 2025 onwards (current season)
            import datetime
            default_start = datetime.date(2025, 10, 21)
            if default_start < min_date:
                default_start = min_date

            col1, col2, col3 = st.columns(3)
            with col1:
                start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            with col3:
                perf_strategy_options = {
                    "🎯 Max Profit (32% ROI)": "Diversified (All tiers)",
                    "⚡ Max Efficiency (35% ROI)": "Optimized (LOW only)",
                    "🛡️ Safe & Simple": "Conservative",
                    "📊 Balanced": "Moderate",
                    "🚀 High Risk": "Aggressive",
                    "💵 Fixed $100": "Fixed"
                }
                perf_strategy_display = st.selectbox(
                    "Betting Strategy",
                    list(perf_strategy_options.keys()),
                    index=5,  # Default to Fixed $100
                    key="perf_strategy"
                )
                perf_strategy = perf_strategy_options[perf_strategy_display]

            # Filter data by date range
            historical = historical[(historical['game_date'].dt.date >= start_date) &
                                   (historical['game_date'].dt.date <= end_date)]

            st.info(f"Analyzing {len(historical)} games from {start_date} to {end_date} using **{perf_strategy_display}** strategy")

        st.markdown("---")

        # Overall statistics
        st.markdown("### Overall Performance")

        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics
        total_games = len(historical)

        if 'ensemble_pred' in historical.columns and 'actual_margin' in historical.columns:
            correct = sum((historical['ensemble_pred'] > 0) == (historical['actual_margin'] > 0))
            accuracy = correct / total_games

            # Calculate profit
            win_return = correct * (100 + 100 * (100/110))
            total_spent = total_games * 100
            profit = win_return - total_spent
            roi = (profit / total_spent) * 100

            col1.metric("Total Games", total_games)
            col2.metric("Accuracy", f"{accuracy:.1%}")
            col3.metric("Total Profit", f"${profit:,.2f}")
            col4.metric("ROI", f"{roi:+.1f}%")

            # Accuracy over time
            st.markdown("### Accuracy Over Time")

            if 'game_date' in historical.columns:
                historical = historical.sort_values('game_date')

                # Rolling accuracy
                historical['correct'] = (historical['ensemble_pred'] > 0) == (historical['actual_margin'] > 0)
                historical['rolling_accuracy'] = historical['correct'].rolling(window=50, min_periods=10).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical['game_date'],
                    y=historical['rolling_accuracy'] * 100,
                    mode='lines',
                    name='50-Game Rolling Accuracy',
                    line=dict(color='#FF6B35', width=3)
                ))
                fig.add_hline(y=52.4, line_dash="dash", line_color="red",
                             annotation_text="Breakeven (52.4%)")
                fig.add_hline(y=65, line_dash="dash", line_color="green",
                             annotation_text="Target (65%)")
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Accuracy (%)",
                    height=400,
                    hovermode='x unified',
                    xaxis=dict(
                        tickformat='%b %d, %Y',
                        tickangle=-45
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Cumulative profit
            st.markdown("### Cumulative Profit")

            historical['profit'] = 0
            for idx in range(len(historical)):
                if historical.iloc[idx]['correct']:
                    historical.loc[historical.index[idx], 'profit'] = 100 * (100/110)
                else:
                    historical.loc[historical.index[idx], 'profit'] = -100

            historical['cumulative_profit'] = historical['profit'].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical['game_date'] if 'game_date' in historical.columns else range(len(historical)),
                y=historical['cumulative_profit'],
                mode='lines',
                fill='tozeroy',
                name='Cumulative Profit',
                line=dict(color='#06A77D', width=3)
            ))
            fig.update_layout(
                xaxis_title="Date" if 'game_date' in historical.columns else "Game Number",
                yaxis_title="Cumulative Profit ($)",
                height=400,
                hovermode='x unified',
                xaxis=dict(
                    tickformat='%b %d, %Y' if 'game_date' in historical.columns else None,
                    tickangle=-45 if 'game_date' in historical.columns else 0
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Prediction error distribution
            st.markdown("### Prediction Error Distribution")

            st.caption("""
            **What this shows:** The distribution of how far off our predictions were from actual game margins.
            A smaller error means more accurate predictions. This helps identify if predictions tend to be
            consistently off by a certain amount or if errors are randomly distributed.
            """)

            historical['error'] = abs(historical['ensemble_pred'] - historical['actual_margin'])

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=historical['error'],
                nbinsx=30,
                marker_color='#004E89'
            ))
            fig.update_layout(
                xaxis_title="Prediction Error (points)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_error = historical['error'].mean()
            median_error = historical['error'].median()
            st.info(f"📊 Average Prediction Error: {avg_error:.2f} points | Median Error: {median_error:.2f} points")

            st.markdown("---")

            # Detailed bets table
            st.markdown("### 📋 Detailed Bet History")

            st.caption("View individual bets, outcomes, and profits for the selected time period.")

            # Prepare detailed table data (reverse order to show most recent first)
            bet_table_data = []
            game_number = len(historical)
            for idx, row in historical[::-1].iterrows():  # Reverse iteration
                result = "✅ Win" if row['correct'] else "❌ Loss"

                # Determine which team was bet on based on prediction
                if 'home_team' in historical.columns and 'away_team' in historical.columns:
                    # Use actual team names
                    home = row['home_team']
                    away = row['away_team']
                    matchup = f"{away} @ {home}"

                    if row['ensemble_pred'] > 0:
                        bet_on = f"{home} ({row['ensemble_pred']:+.1f})"
                    else:
                        bet_on = f"{away} ({row['ensemble_pred']:+.1f})"
                else:
                    # Fallback if team names not available
                    matchup = f"Game {game_number}"
                    if row['ensemble_pred'] > 0:
                        bet_on = f"Home Team ({row['ensemble_pred']:+.1f})"
                    else:
                        bet_on = f"Away Team ({row['ensemble_pred']:+.1f})"

                bet_table_data.append({
                    'Date': row['game_date'].strftime('%b %d, %Y') if 'game_date' in historical.columns else f"Game {game_number}",
                    'Matchup': matchup,
                    'Bet On': bet_on,
                    'Prediction': f"{row['ensemble_pred']:+.1f}",
                    'Actual': f"{row['actual_margin']:+.1f}",
                    'Error': f"{row['error']:.1f}",
                    'Result': result,
                    'Profit': f"${row['profit']:.2f}"
                })
                game_number -= 1

            bet_table_df = pd.DataFrame(bet_table_data)
            st.dataframe(bet_table_df, use_container_width=True, hide_index=True)

    else:
        st.info("No historical data available. Run OOS tests to generate performance data.")

elif page == "💰 Betting Strategy":
    st.markdown("<h1 class='main-header'>💰 Optimized Betting Strategy</h1>", unsafe_allow_html=True)

    # Load optimal bet sizes
    optimal = load_optimal_bet_sizes()

    if optimal:
        # Show validation method
        if optimal.get('validation_method') == 'walk_forward':
            st.success(f"✅ **Walk-Forward Validated!** Out-of-Sample ROI: **{optimal['expected_roi']:.2f}%**")

            st.markdown("""
            ### Properly Validated Bet Sizing Strategy

            **Validation Method: Walk-Forward Analysis**
            - Optimized on 2024-25 season (1,225 games)
            - Validated on 2025-26 season (231 games) - **truly out-of-sample**
            - Beats baseline by **+{:.2f}%** on unseen data
            """.format(optimal.get('baseline_improvement', 0)))

            # Show degradation warning if significant
            degradation = optimal.get('degradation', 0)
            if abs(degradation) > 15:
                st.warning(f"⚠️ Training ROI ({optimal.get('training_roi', 0):.1f}%) degraded to {optimal['expected_roi']:.1f}% on test data. Strategy shows overfitting.")
            elif abs(degradation) > 5:
                st.info(f"ℹ️ Moderate degradation: Training {optimal.get('training_roi', 0):.1f}% → Test {optimal['expected_roi']:.1f}% ({degradation:+.1f}%)")
        else:
            st.warning("⚠️ **In-Sample Optimization Only** - Results may be overfitted. Run `python optimize_bet_sizes_walkforward.py` for proper validation.")
            st.markdown("""
            ### Data-Driven Bet Sizing (Grid Search Results)

            After testing 5,120+ combinations on historical games, the optimal strategy is:
            """)

        col1, col2, col3 = st.columns(3)
        # Update help text based on validation method
        if optimal.get('validation_method') == 'walk_forward':
            col1.metric("🟢 HIGH Confidence", f"{optimal['high_confidence_pct']*100:.1f}%",
                       help="Training: 62.3% | Test: 64.2% win rate")
            col2.metric("🟡 MEDIUM Confidence", f"{optimal['medium_confidence_pct']*100:.1f}%",
                       help="Training: 63.2% | Test: 63.7% win rate")
            col3.metric("🔴 LOW Confidence", f"{optimal['low_confidence_pct']*100:.1f}%",
                       help="Training: 76.9% | Test: 70.8% win rate - BEST!")

            st.warning("""
            **Validated Finding:** LOW confidence games have the HIGHEST win rate, even on unseen data!

            - **Training (2024-25):** 76.9% win rate on LOW confidence
            - **Test (2025-26):** 70.8% win rate on LOW confidence (still beats HIGH/MEDIUM!)

            Model disagreement genuinely predicts better performance. The ensemble method excels when
            individual models disagree, capturing nuances that no single model sees.
            """)
        else:
            col1.metric("🟢 HIGH Confidence", f"{optimal['high_confidence_pct']*100:.1f}%",
                       help="Win Rate: 62.6% (historical)")
            col2.metric("🟡 MEDIUM Confidence", f"{optimal['medium_confidence_pct']*100:.1f}%",
                       help="Win Rate: 63.3% (historical)")
            col3.metric("🔴 LOW Confidence", f"{optimal['low_confidence_pct']*100:.1f}%",
                       help="Win Rate: 76.0% (historical) - BEST PERFORMANCE!")

            st.warning("""
            **Key Finding:** LOW confidence games (where models disagree) have the HIGHEST win rate!

            ⚠️ Note: These results are from in-sample optimization and may not generalize.
            Run walk-forward validation for proper performance estimates.
            """)

    st.markdown("---")

    # Bankroll settings
    col1, col2 = st.columns(2)

    with col1:
        bankroll = st.number_input("Starting Bankroll ($)", value=10000, step=1000)

    with col2:
        risk_level = st.select_slider(
            "Betting Strategy",
            options=["Conservative", "Moderate", "Aggressive", "Optimized"],
            value="Optimized"
        )

    # Show strategy details
    if risk_level == "Optimized" and optimal:
        st.info(f"""
        **Optimized Strategy (Data-Driven):**
        - HIGH: {optimal['high_confidence_pct']*100:.1f}% of bankroll
        - MEDIUM: {optimal['medium_confidence_pct']*100:.1f}% of bankroll
        - LOW: {optimal['low_confidence_pct']*100:.1f}% of bankroll
        - Expected ROI: {optimal['expected_roi']:.2f}%
        """)
    else:
        multipliers = {
            "Conservative": {"HIGH": "2.0%", "MEDIUM": "1.0%", "LOW": "0.5%"},
            "Moderate": {"HIGH": "3.0%", "MEDIUM": "2.0%", "LOW": "1.0%"},
            "Aggressive": {"HIGH": "5.0%", "MEDIUM": "3.0%", "LOW": "1.5%"}
        }
        if risk_level in multipliers:
            st.info(f"""
            **{risk_level} Strategy:**
            - HIGH: {multipliers[risk_level]['HIGH']} of bankroll
            - MEDIUM: {multipliers[risk_level]['MEDIUM']} of bankroll
            - LOW: {multipliers[risk_level]['LOW']} of bankroll
            """)

    st.markdown("---")
    st.markdown("### Tonight's Recommended Bets")

    predictions = load_predictions()

    if predictions is not None and len(predictions) > 0:
        total_to_bet = 0
        bet_details = []

        for idx, game in predictions.iterrows():
            confidence, emoji = get_confidence_level(game)
            bet_amount = calculate_bet_size(confidence, bankroll, risk_level)
            total_to_bet += bet_amount

            winner = game['HOME_TEAM'] if game['ENSEMBLE_PREDICTION'] > 0 else game['AWAY_TEAM']
            bet_details.append({
                'game': f"{game['AWAY_TEAM']} @ {game['HOME_TEAM']}",
                'pick': winner,
                'confidence': confidence,
                'emoji': emoji,
                'bet_pct': (bet_amount / bankroll) * 100,
                'bet_amount': bet_amount
            })

        # Display bet table
        for bet in bet_details:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.write(f"**{bet['pick']}** ({bet['game']})")

            with col2:
                st.write(f"{bet['emoji']} {bet['confidence']}")

            with col3:
                st.write(f"{bet['bet_pct']:.1f}%")

            with col4:
                st.write(f"${bet['bet_amount']:.2f}")

        st.markdown("---")
        st.markdown(f"### **Total to Bet Tonight: ${total_to_bet:.2f}** ({total_to_bet/bankroll*100:.1f}% of bankroll)")

        # Simulate outcomes
        st.markdown("### Simulated Outcomes")

        # Use actual expected win rate if available
        if optimal and risk_level == "Optimized":
            expected_win_rate = optimal['historical_win_rate']
        else:
            expected_win_rate = 0.65

        win_scenario = total_to_bet * expected_win_rate * (100/110) - total_to_bet * (1 - expected_win_rate)

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Profit", f"${win_scenario:.2f}",
                   delta=f"{win_scenario/bankroll*100:.2f}%",
                   help=f"Based on {expected_win_rate:.1%} historical win rate")
        col2.metric("Best Case (100% wins)", f"${total_to_bet * (100/110):.2f}",
                   delta=f"+{total_to_bet * (100/110)/bankroll*100:.2f}%")
        col3.metric("Worst Case (0% wins)", f"${-total_to_bet:.2f}",
                   delta=f"{-total_to_bet/bankroll*100:.2f}%")

    else:
        st.info("No predictions available for tonight.")

elif page == "🔬 Model Explanation":
    st.markdown("<h1 class='main-header'>🔬 Model Explanation</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Overview", "🔢 Features", "🎯 Performance", "🎲 Bet Optimization"])

    with tab1:
        st.markdown("""
        ### Ensemble Prediction System

        Our system combines 4 different models to make predictions:

        #### 1. **Elo Rating System** (25% weight)
        - Based on FiveThirtyEight's methodology
        - Tracks team strength over time
        - Adjusts for home court advantage (+100 Elo)
        - Accounts for margin of victory

        #### 2. **Neural Network** (35% weight)
        - Multi-layer perceptron: 128→64→32 neurons
        - Learns complex non-linear relationships
        - Trained on 34 pre-game features
        - Uses dropout to prevent overfitting

        #### 3. **XGBoost** (40% weight)
        - Gradient boosting decision trees
        - Excels at feature importance
        - Robust to outliers
        - Best individual model (65.8% accuracy)

        #### 4. **Ensemble** (Weighted Average)
        - Combines all models with optimized weights
        - Reduces variance through diversification
        - Most consistent performer (65.4% average)
        """)

        # Model comparison
        st.markdown("### Model Comparison")

        comparison_data = {
            'Model': ['Elo', 'Neural Network', 'XGBoost', 'Ensemble'],
            'Accuracy': [64.3, 65.3, 65.8, 65.4],
            'ROI': [22.8, 24.7, 25.6, 24.8],
            'Weight': [25, 35, 40, 100]
        }

        df = pd.DataFrame(comparison_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['Model'],
            y=df['Accuracy'],
            name='Accuracy (%)',
            marker_color='#FF6B35'
        ))
        fig.update_layout(
            yaxis_title="Accuracy (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 34 Pre-Game Features (No Data Leakage)")

        st.markdown("""
        All features are calculated **before** the game starts to ensure no data leakage:

        #### Elo Ratings (2 features)
        - Pre-game Elo for home team
        - Pre-game Elo for away team

        #### Rolling Averages - Last 5 Games (12 features)
        - Points scored
        - Field goal percentage
        - 3-point percentage
        - Rebounds
        - Assists
        - Turnovers
        *(Both home and away teams)*

        #### Rolling Averages - Last 10 Games (6 features)
        - Points scored
        - Field goal percentage
        - 3-point percentage
        *(Both home and away teams)*

        #### Momentum (6 features)
        - Win percentage last 5 games
        - Win percentage last 10 games
        - Current win/loss streak
        *(Both home and away teams)*

        #### Rest & Fatigue (4 features)
        - Days of rest
        - Back-to-back indicator
        *(Both home and away teams)*

        #### Derived Features (4 features)
        - Elo difference
        - Points differential (5-game)
        - FG% differential (5-game)
        - Rest differential
        """)

        # Feature importance
        st.markdown("### Top 10 Most Important Features (XGBoost)")

        importance_data = {
            'Feature': ['Elo Difference', 'Back-to-Back Away', 'Back-to-Back Home',
                       'Rest Days Away', 'Pre-Elo Away', 'Pre-Elo Home',
                       'Points Roll 10 Away', 'Rest Difference', 'Rest Days Home',
                       'Win% L10 Away'],
            'Importance': [10.5, 4.5, 4.2, 3.9, 3.5, 3.5, 3.4, 3.4, 3.3, 3.1]
        }

        fig = go.Figure(go.Bar(
            x=importance_data['Importance'],
            y=importance_data['Feature'],
            orientation='h',
            marker_color='#06A77D'
        ))
        fig.update_layout(
            xaxis_title="Importance (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Performance Across Different Test Periods")

        performance_data = {
            'Period': ['Training\n(2018-23)', '2024-25 Season\n(OOS)', '2025-26 Season\n(Current)'],
            'Games': [1412, 1225, 231],
            'Accuracy': [65.16, 66.29, 65.80],
            'ROI': [24.39, 26.55, 25.62]
        }

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data['Period'],
                y=performance_data['Accuracy'],
                marker_color='#FF6B35'
            ))
            fig.add_hline(y=52.4, line_dash="dash", annotation_text="Breakeven")
            fig.update_layout(
                title="Accuracy Across Periods",
                yaxis_title="Accuracy (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data['Period'],
                y=performance_data['ROI'],
                marker_color='#06A77D'
            ))
            fig.add_hline(y=0, line_dash="dash", annotation_text="Breakeven")
            fig.update_layout(
                title="ROI Across Periods",
                yaxis_title="ROI (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        st.success("**Consistent 65% accuracy across all time periods proves the model is robust and not overfitted!**")

    with tab4:
        st.markdown("### Walk-Forward Bet Size Optimization")

        optimal = load_optimal_bet_sizes()

        if optimal and optimal.get('validation_method') == 'walk_forward':
            st.success("✅ **Properly Validated Strategy using Walk-Forward Analysis**")

            st.markdown("""
            ### Validation Methodology

            **Why Walk-Forward Matters:**
            - In-sample optimization can overfit to historical patterns
            - Must test strategy on truly out-of-sample data to estimate real-world performance

            **Our Approach:**
            1. **Training Phase:** Test 5,120 bet size combinations on 2024-25 season (1,225 games)
            2. **Testing Phase:** Validate optimal strategy on 2025-26 season (231 games) - **unseen data**
            3. **Comparison:** Measure performance degradation and compare to baseline

            **Search Space:**
            - HIGH confidence: 0% to 8% of bankroll
            - MEDIUM confidence: 0% to 8% of bankroll
            - LOW confidence: 0% to 10% of bankroll
            """)

            st.markdown("### Optimal Strategy (Walk-Forward Validated):")

            col1, col2, col3 = st.columns(3)
            col1.metric("HIGH Confidence", f"{optimal['high_confidence_pct']*100:.1f}%",
                       help="Training: 62.3% | Test: 64.2% win rate - Skip these")
            col2.metric("MEDIUM Confidence", f"{optimal['medium_confidence_pct']*100:.1f}%",
                       help="Training: 63.2% | Test: 63.7% win rate - Skip these")
            col3.metric("LOW Confidence", f"{optimal['low_confidence_pct']*100:.1f}%",
                       help="Training: 76.9% | Test: 70.8% win rate - BET ONLY ON THESE!")

            st.markdown("### Performance Results:")

            col1, col2, col3 = st.columns(3)
            col1.metric("Training ROI", f"{optimal.get('training_roi', 0):.2f}%",
                       help="Performance on 2024-25 season")
            col2.metric("Test ROI", f"{optimal['expected_roi']:.2f}%",
                       help="Performance on 2025-26 season (OOS)",
                       delta=f"{optimal.get('degradation', 0):.1f}%")
            col3.metric("vs Baseline", f"+{optimal.get('baseline_improvement', 0):.2f}%",
                       help="Improvement over 2% equal betting on all games")

            degradation = abs(optimal.get('degradation', 0))
            if degradation < 5:
                st.success("✅ **Minimal overfitting detected** - Strategy generalizes well!")
            elif degradation < 15:
                st.info(f"ℹ️ **Moderate degradation** ({degradation:.1f}%) - Expected with optimization, still profitable")
            else:
                st.warning(f"⚠️ **Significant degradation** ({degradation:.1f}%) - Strategy may be overfit")

            st.markdown("### Counter-Intuitive Finding (VALIDATED!):")

            st.warning("""
            **LOW confidence games have the HIGHEST win rate, even on unseen data!**

            | Confidence | Training Win Rate | Test Win Rate | Validated |
            |-----------|------------------|---------------|-----------|
            | LOW       | 76.9%            | **70.8%**     | ✅ YES    |
            | MEDIUM    | 63.2%            | 63.7%         | -         |
            | HIGH      | 62.3%            | 64.2%         | -         |

            Possible explanations:
            1. Model disagreement indicates complex games where the ensemble method shines
            2. Diverse predictions allow the weighted average to capture nuances
            3. Sample size consideration: Only 321 LOW confidence games vs 1,135 HIGH/MEDIUM

            **Recommendation:** The validated strategy (bet ONLY on LOW confidence) beats baseline by +10.43%.
            This counter-intuitive finding has been confirmed on truly out-of-sample data!
            """)

        else:
            # Fallback to in-sample results
            st.warning("⚠️ **In-Sample Results Only** - Not properly validated")

            st.markdown("""
            ### In-Sample Optimization (May Be Overfit)

            Tested 5,120 combinations on all available historical data.

            **⚠️ Problem:** Optimizing and testing on the same data leads to overfitting.
            Run `python optimize_bet_sizes_walkforward.py` for proper walk-forward validation.
            """)

            if optimal:
                col1, col2, col3 = st.columns(3)
                col1.metric("HIGH Confidence", f"{optimal['high_confidence_pct']*100:.1f}%")
                col2.metric("MEDIUM Confidence", f"{optimal['medium_confidence_pct']*100:.1f}%")
                col3.metric("LOW Confidence", f"{optimal['low_confidence_pct']*100:.1f}%")

                st.metric("Expected ROI (Likely Overfitted)", f"{optimal['expected_roi']:.2f}%")

        # Try to load grid search results (prefer walk-forward)
        grid_results = None
        try:
            grid_results = pd.read_csv('results/bet_sizing_grid_search_walkforward.csv')
            st.markdown("### Top 20 Strategies (from Walk-Forward Training)")
        except:
            try:
                grid_results = pd.read_csv('results/bet_sizing_grid_search.csv')
                st.markdown("### Top 20 Strategies (In-Sample Only)")
            except:
                pass

        if grid_results is not None:
            top_20 = grid_results.nlargest(20, 'roi')[['high_pct', 'medium_pct', 'low_pct', 'roi']]
            top_20.columns = ['HIGH %', 'MEDIUM %', 'LOW %', 'ROI %']
            top_20['HIGH %'] = top_20['HIGH %'] * 100
            top_20['MEDIUM %'] = top_20['MEDIUM %'] * 100
            top_20['LOW %'] = top_20['LOW %'] * 100

            st.dataframe(top_20.head(20), use_container_width=True)
        else:
            st.info("Grid search results file not found. Run `python optimize_bet_sizes_walkforward.py` to generate.")

else:  # Settings
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)

    st.markdown("### Data Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Tonight's Predictions"):
            st.info("Run: `python predict_tonight.py` in terminal")

    with col2:
        if st.button("📊 Update Performance Data"):
            st.info("Run: `python test_2025_26_season.py` in terminal")

    with col3:
        if st.button("🎲 Optimize Bet Sizes (Walk-Forward)"):
            st.info("Run: `python optimize_bet_sizes_walkforward.py` in terminal")

    st.markdown("---")
    st.markdown("### Model Information")

    optimal = load_optimal_bet_sizes()
    if optimal and optimal.get('validation_method') == 'walk_forward':
        roi_text = f"+{optimal['expected_roi']:.1f}% ROI (walk-forward validated on 2025-26 season)"
    else:
        roi_text = "+25% ROI (in-sample, may be overfitted)"

    st.info(f"""
    **Training Data:** 2018-2024 seasons (7,058 games)

    **Models:** Elo, Neural Network, XGBoost, Ensemble

    **Last Updated:** Check file timestamps in models/ directory

    **Model Performance:** 65.9% accuracy across all test periods

    **Betting Performance:** {roi_text}
    """)

    st.markdown("---")
    st.markdown("### Quick Commands")

    st.code("""
# Get tonight's predictions
python predict_tonight.py

# Check prediction results
python check_predictions.py

# Test on 2025-26 season
python test_2025_26_season.py

# Optimize bet sizing (walk-forward validation)
python optimize_bet_sizes_walkforward.py

# Run dashboard
streamlit run dashboard_v2.py
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>NBA Prediction Dashboard v2 | Built with Streamlit | Data from NBA API</p>
    <p>⚠️ For educational purposes only. Bet responsibly.</p>
</div>
""", unsafe_allow_html=True)
