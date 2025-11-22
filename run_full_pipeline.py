"""
Full Pipeline - Runs entire NBA prediction model workflow
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"  {description}")
    print("="*70)

    try:
        result = subprocess.run(
            [sys.executable, command],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error:")
        print(f"  {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  NBA PREDICTION MODEL - FULL PIPELINE")
    print("  Point Spread Prediction System")
    print("="*70)

    print("\nThis will:")
    print("  1. Collect NBA game data (2018-2024)")
    print("  2. Download FiveThirtyEight historical data")
    print("  3. Build features (Elo, rolling stats, momentum)")
    print("  4. Train Neural Network, XGBoost, and Ensemble models")
    print("  5. Evaluate performance and simulate betting ROI")
    print("\nEstimated time: 20-30 minutes")

    input("\nPress Enter to continue or Ctrl+C to cancel...")

    # Step 1: Data Collection
    if not run_command('run_data_collection.py', 'Step 1/2: Data Collection'):
        print("\n✗ Pipeline failed at data collection stage")
        return

    # Step 2: Model Training
    if not run_command('run_model_training.py', 'Step 2/2: Model Training & Evaluation'):
        print("\n✗ Pipeline failed at model training stage")
        return

    # Success!
    print("\n" + "="*70)
    print("  ✓ PIPELINE COMPLETE!")
    print("="*70)

    print("\nResults:")
    print("  📊 Predictions: results/predictions.csv")
    print("  📈 Model Comparison: results/model_comparison.csv")
    print("  🤖 Saved Models: models/")

    print("\nNext Steps:")
    print("  - Review results/model_comparison.csv for performance metrics")
    print("  - Check betting ROI in console output above")
    print("  - Use trained models to predict upcoming games")

    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user")
        sys.exit(0)
