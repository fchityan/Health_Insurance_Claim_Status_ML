from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .pipeline import (
    denied_probability,
    evaluate_predictions,
    load_and_prepare_data,
    split_data,
    threshold_predict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked claim status model.")
    parser.add_argument("--data-dir", default=".", help="Directory containing source CSV files.")
    parser.add_argument(
        "--model-path",
        default="artifacts/best_model.joblib",
        help="Path to saved model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/test_metrics.json",
        help="Path to test metrics JSON.",
    )
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    random_state = int(artifact.get("random_state", 42))

    X, y = load_and_prepare_data(args.data_dir)
    splits = split_data(X, y, random_state=random_state)

    denied_probs = denied_probability(model, splits.X_test)
    y_pred = threshold_predict(model, splits.X_test, threshold)
    metrics = evaluate_predictions(splits.y_test, y_pred, denied_probs)
    metrics["threshold"] = threshold
    metrics["model_name"] = artifact.get("model_name", "unknown")

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "test_classification_report.txt").write_text(
        metrics["classification_report"], encoding="utf-8"
    )

    predictions = pd.DataFrame(
        {
            "y_true": splits.y_test.reset_index(drop=True),
            "denied_probability": denied_probs,
            "y_pred": y_pred,
        }
    )
    predictions.to_csv(outputs_dir / "test_predictions.csv", index=False)

    printable = {k: v for k, v in metrics.items() if k != "classification_report"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
