from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ARTIFACT_PATH, DEFAULT_PREDICTIONS_PATH
from src.inference.predict import load_artifact, predict_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch churn predictions with a trained artifact.")
    parser.add_argument("--input", required=True, help="Path to the CSV file to score.")
    parser.add_argument("--output", default=str(DEFAULT_PREDICTIONS_PATH), help="Path to save predictions.")
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH), help="Path to the saved model artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.artifact)
    dataframe = pd.read_csv(args.input)
    predictions = predict_dataframe(dataframe, artifact)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
