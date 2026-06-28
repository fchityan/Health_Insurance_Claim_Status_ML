from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from .pipeline import (
    build_preprocessor,
    evaluate_predictions,
    load_and_prepare_data,
    split_data,
    threshold_predict,
)


def train_one_model(name: str, classifier, preprocessor, X_train, y_train, X_val, y_val, threshold: float):
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = threshold_predict(model, X_val, threshold=threshold)
    metrics = evaluate_predictions(y_val, y_pred)
    return name, model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train claim status models and save the best one.")
    parser.add_argument("--data-dir", default=".", help="Directory that contains source CSV files.")
    parser.add_argument(
        "--model-out",
        default="artifacts/best_model.joblib",
        help="Output path for the saved model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/validation_metrics.json",
        help="Output path for validation metrics JSON.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory for generated training outputs (summaries and reports).",
    )
    parser.add_argument("--threshold", type=float, default=0.40, help="DENIED probability threshold.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible splits.")

    args = parser.parse_args()

    X, y = load_and_prepare_data(args.data_dir)
    splits = split_data(X, y, random_state=args.random_state)
    preprocessor = build_preprocessor(splits.X_train)

    candidates = [
        (
            "random_forest",
            RandomForestClassifier(n_estimators=100, random_state=args.random_state),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(n_estimators=100, random_state=args.random_state),
        ),
    ]

    results = []
    for model_name, classifier in candidates:
        results.append(
            train_one_model(
                model_name,
                classifier,
                preprocessor,
                splits.X_train,
                splits.y_train,
                splits.X_val,
                splits.y_val,
                args.threshold,
            )
        )

    # Select best model by DENIED recall, then DENIED F1 for tie-break.
    best_name, best_model, best_metrics = sorted(
        results,
        key=lambda r: (r[2]["denied_recall"], r[2]["denied_f1"]),
        reverse=True,
    )[0]

    artifact = {
        "model_name": best_name,
        "threshold": args.threshold,
        "random_state": args.random_state,
        "feature_columns": list(splits.X_train.columns),
        "model": best_model,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        name: metrics for name, _, metrics in results
    }
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "best_model": best_name,
        "threshold": args.threshold,
        "random_state": args.random_state,
        "validation": {
            name: {
                "denied_precision": metrics["denied_precision"],
                "denied_recall": metrics["denied_recall"],
                "denied_f1": metrics["denied_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            for name, _, metrics in results
        },
    }
    with (outputs_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_rows = []
    for name, _, metrics in results:
        report_rows.append(
            {
                "model": name,
                "denied_precision": metrics["denied_precision"],
                "denied_recall": metrics["denied_recall"],
                "denied_f1": metrics["denied_f1"],
            }
        )
    pd.DataFrame(report_rows).to_csv(outputs_dir / "validation_model_comparison.csv", index=False)

    for name, _, metrics in results:
        with (outputs_dir / f"validation_report_{name}.txt").open("w", encoding="utf-8") as f:
            f.write(metrics["classification_report"])

    print(f"Saved best model: {best_name}")
    print(f"Model path: {model_out}")
    print(f"Metrics path: {metrics_out}")
    print(f"Outputs directory: {outputs_dir}")
    print("Validation metrics for best model:")
    print(best_metrics["classification_report"])


if __name__ == "__main__":
    main()
