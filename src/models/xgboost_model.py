"""
XGBoost Model for NBA Point Spread Prediction
Gradient boosting for margin of victory prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import os

class NBAXGBoost:
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=200,
                 subsample=0.8, colsample_bytree=0.8):
        """
        Initialize XGBoost model

        Parameters:
        - max_depth: Maximum tree depth
        - learning_rate: Learning rate (eta)
        - n_estimators: Number of boosting rounds
        - subsample: Fraction of samples per tree
        - colsample_bytree: Fraction of features per tree
        """
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.model = None
        self.feature_importance = None

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train XGBoost model

        Parameters:
        - X_train: Training features
        - y_train: Training targets (point differential)
        - X_val: Validation features (optional)
        - y_val: Validation targets (optional)
        - verbose: Print training progress
        """
        print("Training XGBoost model...")

        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=verbose
            )
        else:
            self.model.fit(X_train, y_train, verbose=verbose)

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        print("Training complete!")

    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """
        Perform grid search for hyperparameter optimization

        Parameters:
        - X_train: Training features
        - y_train: Training targets
        - cv: Number of cross-validation folds
        """
        print("Optimizing hyperparameters via grid search...")

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MAE: {-grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.params.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        self.feature_importance = self.model.feature_importances_

        return grid_search.best_params_

    def predict(self, X):
        """
        Predict point spread (margin of victory)

        Returns:
        - Predicted point differential (positive = home team favored)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(X)

    def predict_winner(self, X):
        """
        Predict game winner

        Returns:
        - 1 if home team predicted to win, 0 if away team
        """
        margins = self.predict(X)
        return (margins > 0).astype(int)

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance rankings

        Parameters:
        - feature_names: List of feature names (optional)

        Returns:
        - DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet!")

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names is not None else range(len(self.feature_importance)),
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_feature_importance(self, top_n=20):
        """Plot top N most important features"""
        try:
            import matplotlib.pyplot as plt

            importance_df = self.get_feature_importance()

            plt.figure(figsize=(10, 8))
            plt.barh(
                range(top_n),
                importance_df['importance'].head(top_n),
                align='center'
            )
            plt.yticks(range(top_n), importance_df['feature'].head(top_n))
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300)
            print("Feature importance plot saved to results/feature_importance.png")

        except ImportError:
            print("matplotlib not available for plotting")

    def save_model(self, model_dir='models/xgboost'):
        """Save trained model"""
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'xgb_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save feature importance
        if self.feature_importance is not None:
            importance_path = os.path.join(model_dir, 'feature_importance.pkl')
            with open(importance_path, 'wb') as f:
                pickle.dump(self.feature_importance, f)

        print(f"Model saved to {model_dir}")

    def load_model(self, model_dir='models/xgboost'):
        """Load trained model"""
        model_path = os.path.join(model_dir, 'xgb_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load feature importance
        importance_path = os.path.join(model_dir, 'feature_importance.pkl')
        if os.path.exists(importance_path):
            with open(importance_path, 'rb') as f:
                self.feature_importance = pickle.load(f)

        print(f"Model loaded from {model_dir}")


if __name__ == "__main__":
    print("XGBoost model initialized for NBA predictions")
