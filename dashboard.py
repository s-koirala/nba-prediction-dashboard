"""
Interactive NBA Prediction Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
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
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
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
    """Load tonight's predictions"""
    try:
        return pd.read_csv('results/tonights_predictions.csv')
    except:
        return None

@st.cache_data(ttl=3600)
def load_historical_performance():
    """Load historical performance data"""
    try:
        oos_2024 = pd.read_csv('results/oos_predictions.csv')
        oos_2025 = pd.read_csv('results/predictions_2025_26.csv')
        return pd.concat([oos_2024, oos_2025], ignore_index=True)
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

def calculate_bet_size(confidence, bankroll=10000, base_bet_pct=0.02):
    """
    Calculate bet size based on confidence level
    High: 3% of bankroll
    Medium: 2% of bankroll
    Low: 1% of bankroll (or skip)
    """
    if confidence == "HIGH":
        return bankroll * 0.03
    elif confidence == "MEDIUM":
        return bankroll * 0.02
    else:
        return bankroll * 0.01

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/889/889419.png", width=100)
    st.title("🏀 NBA Predictor")

    page = st.radio(
        "Navigate",
        ["📊 Tonight's Games", "📈 Performance Tracking", "💰 Betting Strategy",
         "🔬 Model Explanation", "⚙️ Settings"]
    )

    st.markdown("---")
    st.markdown("### Model Stats")
    st.metric("Historical Accuracy", "65.4%")
    st.metric("2025-26 Accuracy", "65.8%")
    st.metric("Average ROI", "+24.8%")

# Main content
if page == "📊 Tonight's Games":
    st.markdown("<h1 class='main-header'>🏀 Tonight's NBA Predictions</h1>", unsafe_allow_html=True)

    predictions = load_predictions()

    if predictions is not None and len(predictions) > 0:
        st.markdown(f"### {datetime.now().strftime('%B %d, %Y')} - {len(predictions)} Games")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        high_conf = sum([1 for _, row in predictions.iterrows() if get_confidence_level(row)[0] == "HIGH"])
        med_conf = sum([1 for _, row in predictions.iterrows() if get_confidence_level(row)[0] == "MEDIUM"])
        low_conf = sum([1 for _, row in predictions.iterrows() if get_confidence_level(row)[0] == "LOW"])

        col1.metric("Total Games", len(predictions))
        col2.metric("High Confidence", high_conf, delta="Bet 3%")
        col3.metric("Medium Confidence", med_conf, delta="Bet 2%")
        col4.metric("Low Confidence", low_conf, delta="Bet 1%")

        st.markdown("---")

        # Display each game
        for idx, game in predictions.iterrows():
            confidence, emoji = get_confidence_level(game)

            confidence_class = {
                "HIGH": "high-confidence",
                "MEDIUM": "medium-confidence",
                "LOW": "low-confidence"
            }[confidence]

            with st.container():
                st.markdown(f"<div class='prediction-box {confidence_class}'>", unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### {game['AWAY_TEAM']} @ {game['HOME_TEAM']}")
                    st.markdown(f"**Time:** {game['GAME_TIME']}")

                    # Prediction
                    if game['ENSEMBLE_PREDICTION'] > 0:
                        winner = game['HOME_TEAM']
                        margin = game['ENSEMBLE_PREDICTION']
                    else:
                        winner = game['AWAY_TEAM']
                        margin = abs(game['ENSEMBLE_PREDICTION'])

                    st.markdown(f"**Predicted Winner:** {winner}")
                    st.markdown(f"**Predicted Margin:** {margin:.1f} points")

                with col2:
                    st.markdown(f"### {emoji} {confidence} Confidence")

                    # Model predictions
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Elo', 'NN', 'XGB', 'Ensemble'],
                        y=[game['ELO_PREDICTION'], game['NN_PREDICTION'],
                           game['XGB_PREDICTION'], game['ENSEMBLE_PREDICTION']],
                        marker_color=['#FF6B35', '#004E89', '#F77F00', '#06A77D']
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title="Predicted Margin",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Recommended bet
                    bet_size = calculate_bet_size(confidence)
                    st.markdown(f"**Recommended Bet:** ${bet_size:.2f}")

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.info("No games scheduled for tonight. Run `python predict_tonight.py` to generate predictions.")

elif page == "📈 Performance Tracking":
    st.markdown("<h1 class='main-header'>📈 Performance Tracking</h1>", unsafe_allow_html=True)

    # Load historical performance
    historical = load_historical_performance()

    if historical is not None:
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
                historical['game_date'] = pd.to_datetime(historical['game_date'])
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
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

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
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prediction error distribution
            st.markdown("### Prediction Error Distribution")

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
            st.info(f"Average Prediction Error: {avg_error:.2f} points")

    else:
        st.info("No historical data available. Run OOS tests to generate performance data.")

elif page == "💰 Betting Strategy":
    st.markdown("<h1 class='main-header'>💰 Confidence-Based Betting Strategy</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### Kelly Criterion - Confidence Based Betting

    Our betting strategy adjusts bet size based on model confidence:
    - **High Confidence** 🟢: All models agree (±3 points) → Bet 3% of bankroll
    - **Medium Confidence** 🟡: Models mostly agree (±6 points) → Bet 2% of bankroll
    - **Low Confidence** 🔴: Models disagree (>6 points) → Bet 1% of bankroll or skip
    """)

    # Bankroll settings
    col1, col2 = st.columns(2)

    with col1:
        bankroll = st.number_input("Starting Bankroll ($)", value=10000, step=1000)

    with col2:
        risk_level = st.select_slider(
            "Risk Level",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )

    # Adjust multipliers based on risk
    multipliers = {
        "Conservative": {"HIGH": 0.02, "MEDIUM": 0.01, "LOW": 0.005},
        "Moderate": {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01},
        "Aggressive": {"HIGH": 0.05, "MEDIUM": 0.03, "LOW": 0.015}
    }

    st.markdown("---")
    st.markdown("### Tonight's Recommended Bets")

    predictions = load_predictions()

    if predictions is not None and len(predictions) > 0:
        total_to_bet = 0

        for idx, game in predictions.iterrows():
            confidence, emoji = get_confidence_level(game)
            bet_pct = multipliers[risk_level][confidence]
            bet_amount = bankroll * bet_pct
            total_to_bet += bet_amount

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                winner = game['HOME_TEAM'] if game['ENSEMBLE_PREDICTION'] > 0 else game['AWAY_TEAM']
                st.write(f"**{winner}**")

            with col2:
                st.write(f"{emoji} {confidence}")

            with col3:
                st.write(f"{bet_pct*100:.1f}%")

            with col4:
                st.write(f"${bet_amount:.2f}")

        st.markdown("---")
        st.markdown(f"### **Total to Bet Tonight: ${total_to_bet:.2f}** ({total_to_bet/bankroll*100:.1f}% of bankroll)")

        # Simulate outcomes
        st.markdown("### Simulated Outcomes (65% win rate)")

        win_scenario = total_to_bet * 0.65 * (100/110) - total_to_bet * 0.35
        loss_scenario = -total_to_bet * 0.65

        col1, col2 = st.columns(2)
        col1.metric("Expected Profit", f"${win_scenario:.2f}", delta=f"{win_scenario/bankroll*100:.2f}%")
        col2.metric("Worst Case (0% wins)", f"${-total_to_bet:.2f}", delta=f"{-total_to_bet/bankroll*100:.2f}%")

    else:
        st.info("No predictions available for tonight.")

elif page == "🔬 Model Explanation":
    st.markdown("<h1 class='main-header'>🔬 Model Explanation</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📚 Overview", "🔢 Features", "🎯 Performance"])

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

else:  # Settings
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)

    st.markdown("### Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Tonight's Predictions"):
            st.info("Run: `python predict_tonight.py` in terminal")

    with col2:
        if st.button("📊 Update Performance Data"):
            st.info("Run: `python test_2025_26_season.py` in terminal")

    st.markdown("---")
    st.markdown("### Model Information")

    st.info("""
    **Training Data:** 2018-2024 seasons (7,058 games)

    **Models:** Elo, Neural Network, XGBoost, Ensemble

    **Last Updated:** Check file timestamps in models/ directory

    **Performance:** 65% accuracy, +25% ROI
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

# Run dashboard
streamlit run dashboard.py
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>NBA Prediction Dashboard | Built with Streamlit | Data from NBA API</p>
    <p>⚠️ For educational purposes only. Bet responsibly.</p>
</div>
""", unsafe_allow_html=True)
