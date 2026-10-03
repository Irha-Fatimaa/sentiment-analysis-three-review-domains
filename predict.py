"""Command-line sentiment prediction using the trained pipeline."""
import argparse
from pathlib import Path
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('text', help='Review to classify')
parser.add_argument('--model', type=Path, default=Path('outputs/sentiment_model.joblib'))
args = parser.parse_args()
model = joblib.load(args.model)  # Load only model files that you created or trust.
probability = model.predict_proba([args.text])[0, 1]
print(f"{'Positive' if probability >= 0.5 else 'Negative'} | positive probability: {probability:.3f}")
