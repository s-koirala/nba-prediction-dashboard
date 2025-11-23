"""
Neural Network Model for NBA Point Spread Prediction
Multi-layer perceptron for regression (margin of victory)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Try importing TensorFlow/Keras, fall back to sklearn if unavailable
try:
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    USE_KERAS = True
except ImportError:
    from sklearn.neural_network import MLPRegressor
    USE_KERAS = False
    print("TensorFlow not available, using sklearn MLPRegressor")


class NBANeuralNetwork:
    def __init__(self, hidden_layers=[128, 64, 32], learning_rate=0.001, dropout_rate=0.3):
        """
        Initialize Neural Network model

        Parameters:
        - hidden_layers: List of neurons per hidden layer
        - learning_rate: Learning rate for optimizer
        - dropout_rate: Dropout rate for regularization
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()
        self.model = None
        self.use_keras = USE_KERAS

    def build_model(self, input_dim):
        """Build the neural network architecture"""
        if self.use_keras:
            model = keras.Sequential()

            # Input layer
            model.add(layers.Input(shape=(input_dim,)))

            # Hidden layers with dropout
            for neurons in self.hidden_layers:
                model.add(layers.Dense(neurons, activation='relu'))
                model.add(layers.Dropout(self.dropout_rate))

            # Output layer (regression - predicting point differential)
            model.add(layers.Dense(1, activation='linear'))

            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            return model
        else:
            # Sklearn MLPRegressor fallback
            return MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_layers),
                activation='relu',
                solver='adam',
                learning_rate_init=self.learning_rate,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )

    def prepare_features(self, X):
        """Normalize features"""
        return self.scaler.transform(X)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the neural network

        Parameters:
        - X_train: Training features
        - y_train: Training targets (point differential)
        - X_val: Validation features
        - y_val: Validation targets
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        """
        print("Training Neural Network...")

        # Fit scaler on training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            # Create validation split
            X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
                X_train_scaled, y_train, test_size=0.15, random_state=42
            )

        # Build model
        input_dim = X_train_scaled.shape[1]
        self.model = self.build_model(input_dim)

        if self.use_keras:
            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )

            # Train
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )

            return history
        else:
            # Sklearn training
            self.model.fit(X_train_scaled, y_train)
            print(f"Training complete. Best validation score: {self.model.best_validation_score_:.4f}")

    def predict(self, X):
        """
        Predict point spread (margin of victory)

        Returns:
        - Predicted point differential (positive = home team favored)
        """
        X_scaled = self.scaler.transform(X)

        if self.use_keras:
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)

        return predictions

    def predict_winner(self, X):
        """
        Predict game winner (binary classification)

        Returns:
        - 1 if home team predicted to win, 0 if away team
        """
        margins = self.predict(X)
        return (margins > 0).astype(int)

    def save_model(self, model_dir='models/neural_network'):
        """Save trained model"""
        os.makedirs(model_dir, exist_ok=True)

        if self.use_keras:
            self.model.save(os.path.join(model_dir, 'nn_model.keras'))
        else:
            with open(os.path.join(model_dir, 'nn_model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)

        # Save scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {model_dir}")

    def load_model(self, model_dir='models/neural_network'):
        """Load trained model"""
        # Check which model file exists and load accordingly
        keras_path = os.path.join(model_dir, 'nn_model.keras')
        pkl_path = os.path.join(model_dir, 'nn_model.pkl')

        if os.path.exists(keras_path) and self.use_keras:
            self.model = keras.models.load_model(keras_path)
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}. Looking for nn_model.keras or nn_model.pkl")

        # Load scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"Model loaded from {model_dir}")


if __name__ == "__main__":
    print("Neural Network model initialized")
    print(f"Using {'Keras/TensorFlow' if USE_KERAS else 'sklearn MLPRegressor'}")
