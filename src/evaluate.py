from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .pipeline import evaluate_predictions, load_and_prepare_data, split_data, threshold_predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model on test split.")
    parser.add_argument("--data-dir", default=".", help="Directory that contains source CSV files.")
    parser.add_argument(
        "--model-path",
        default="artifacts/best_model.joblib",
        help="Path to saved model artifact from training.",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/test_metrics.json",
        help="Path to write test metrics JSON.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory for generated evaluation outputs.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.40))
    random_state = int(artifact.get("random_state", 42))

    X, y = load_and_prepare_data(args.data_dir)
    splits = split_data(X, y, random_state=random_state)

    y_pred = threshold_predict(model, splits.X_test, threshold=threshold)
    metrics = evaluate_predictions(splits.y_test, y_pred)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with (outputs_dir / "test_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report"])

    predictions_df = pd.DataFrame(
        {
            "y_true": splits.y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
        }
    )
    predictions_df.to_csv(outputs_dir / "test_predictions.csv", index=False)

    print(f"Evaluated model: {artifact.get('model_name', 'unknown')}")
    print(f"Threshold: {threshold}")
    print(f"Test metrics path: {metrics_out}")
    print(f"Outputs directory: {outputs_dir}")
    print("Test classification report:")
    print(metrics["classification_report"])
    print(f"Confusion matrix [DENIED, PAID]: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
