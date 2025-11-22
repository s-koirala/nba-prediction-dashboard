"""
Ensemble Model for NBA Point Spread Prediction
Combines Elo, Neural Network, and XGBoost predictions
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import pickle
import os

class NBAEnsemble:
    def __init__(self, elo_weight=0.25, nn_weight=0.35, xgb_weight=0.40, method='weighted'):
        """
        Initialize Ensemble model

        Parameters:
        - elo_weight: Weight for Elo predictions
        - nn_weight: Weight for Neural Network predictions
        - xgb_weight: Weight for XGBoost predictions
        - method: 'weighted' for weighted average, 'stacking' for meta-learner
        """
        self.elo_weight = elo_weight
        self.nn_weight = nn_weight
        self.xgb_weight = xgb_weight
        self.method = method
        self.meta_model = None

        # Normalize weights
        total = elo_weight + nn_weight + xgb_weight
        self.elo_weight /= total
        self.nn_weight /= total
        self.xgb_weight /= total

    def weighted_average(self, elo_pred, nn_pred, xgb_pred):
        """
        Combine predictions using weighted average

        Parameters:
        - elo_pred: Elo model predictions
        - nn_pred: Neural network predictions
        - xgb_pred: XGBoost predictions

        Returns:
        - Combined predictions
        """
        return (
            self.elo_weight * elo_pred +
            self.nn_weight * nn_pred +
            self.xgb_weight * xgb_pred
        )

    def train_stacking(self, elo_pred, nn_pred, xgb_pred, y_true):
        """
        Train stacking meta-learner

        Parameters:
        - elo_pred: Elo predictions on training set
        - nn_pred: NN predictions on training set
        - xgb_pred: XGBoost predictions on training set
        - y_true: True labels
        """
        print("Training stacking meta-model...")

        # Create feature matrix from base model predictions
        X_meta = np.column_stack([elo_pred, nn_pred, xgb_pred])

        # Train Ridge regression as meta-learner
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(X_meta, y_true)

        print(f"Meta-model coefficients: Elo={self.meta_model.coef_[0]:.3f}, "
              f"NN={self.meta_model.coef_[1]:.3f}, XGB={self.meta_model.coef_[2]:.3f}")

    def predict(self, elo_pred, nn_pred, xgb_pred):
        """
        Make ensemble prediction

        Parameters:
        - elo_pred: Elo model predictions
        - nn_pred: Neural network predictions
        - xgb_pred: XGBoost predictions

        Returns:
        - Ensemble predictions (point differential)
        """
        if self.method == 'weighted':
            return self.weighted_average(elo_pred, nn_pred, xgb_pred)
        elif self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Stacking model not trained yet!")

            X_meta = np.column_stack([elo_pred, nn_pred, xgb_pred])
            return self.meta_model.predict(X_meta)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_winner(self, elo_pred, nn_pred, xgb_pred):
        """
        Predict game winner

        Returns:
        - 1 if home team predicted to win, 0 if away team
        """
        margins = self.predict(elo_pred, nn_pred, xgb_pred)
        return (margins > 0).astype(int)

    def predict_spread_cover(self, elo_pred, nn_pred, xgb_pred, spread_line):
        """
        Predict if home team will cover the spread

        Parameters:
        - spread_line: Vegas spread line (negative = home team favored)

        Returns:
        - 1 if home team covers, 0 if away team covers
        """
        predicted_margins = self.predict(elo_pred, nn_pred, xgb_pred)

        # Home team covers if predicted margin > spread line
        # E.g., if spread is -5 (home favored by 5), need to win by >5
        return (predicted_margins > spread_line).astype(int)

    def get_model_contributions(self, elo_pred, nn_pred, xgb_pred):
        """
        Get individual model contributions to final prediction

        Returns:
        - DataFrame showing each model's contribution
        """
        ensemble_pred = self.predict(elo_pred, nn_pred, xgb_pred)

        df = pd.DataFrame({
            'elo_prediction': elo_pred,
            'nn_prediction': nn_pred,
            'xgb_prediction': xgb_pred,
            'ensemble_prediction': ensemble_pred
        })

        if self.method == 'weighted':
            df['elo_contribution'] = elo_pred * self.elo_weight
            df['nn_contribution'] = nn_pred * self.nn_weight
            df['xgb_contribution'] = xgb_pred * self.xgb_weight

        return df

    def analyze_prediction_agreement(self, elo_pred, nn_pred, xgb_pred):
        """
        Analyze agreement between models

        Returns:
        - Statistics on model agreement
        """
        # Convert to win/loss predictions
        elo_winners = (elo_pred > 0).astype(int)
        nn_winners = (nn_pred > 0).astype(int)
        xgb_winners = (xgb_pred > 0).astype(int)

        # Count agreement
        all_agree = (elo_winners == nn_winners) & (nn_winners == xgb_winners)
        two_agree = (
            ((elo_winners == nn_winners) & (nn_winners != xgb_winners)) |
            ((elo_winners == xgb_winners) & (elo_winners != nn_winners)) |
            ((nn_winners == xgb_winners) & (nn_winners != elo_winners))
        )

        agreement_stats = {
            'all_models_agree': all_agree.sum() / len(all_agree),
            'two_models_agree': two_agree.sum() / len(two_agree),
            'no_agreement': (~all_agree & ~two_agree).sum() / len(all_agree),
            'avg_elo_margin': np.abs(elo_pred).mean(),
            'avg_nn_margin': np.abs(nn_pred).mean(),
            'avg_xgb_margin': np.abs(xgb_pred).mean()
        }

        return agreement_stats

    def save_model(self, model_dir='models/ensemble'):
        """Save ensemble configuration"""
        os.makedirs(model_dir, exist_ok=True)

        config = {
            'elo_weight': self.elo_weight,
            'nn_weight': self.nn_weight,
            'xgb_weight': self.xgb_weight,
            'method': self.method
        }

        with open(os.path.join(model_dir, 'ensemble_config.pkl'), 'wb') as f:
            pickle.dump(config, f)

        if self.meta_model is not None:
            with open(os.path.join(model_dir, 'meta_model.pkl'), 'wb') as f:
                pickle.dump(self.meta_model, f)

        print(f"Ensemble configuration saved to {model_dir}")

    def load_model(self, model_dir='models/ensemble'):
        """Load ensemble configuration"""
        with open(os.path.join(model_dir, 'ensemble_config.pkl'), 'rb') as f:
            config = pickle.load(f)

        self.elo_weight = config['elo_weight']
        self.nn_weight = config['nn_weight']
        self.xgb_weight = config['xgb_weight']
        self.method = config['method']

        meta_path = os.path.join(model_dir, 'meta_model.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.meta_model = pickle.load(f)

        print(f"Ensemble configuration loaded from {model_dir}")


if __name__ == "__main__":
    print("Ensemble model initialized for NBA predictions")
