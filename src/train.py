from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .pipeline import (
    build_preprocessor,
    denied_probability,
    evaluate_predictions,
    load_and_prepare_data,
    select_threshold,
    split_data,
)


def parse_threshold(value: str) -> str | float:
    if value.lower() == "auto":
        return "auto"
    threshold = float(value)
    if not 0 < threshold < 1:
        raise argparse.ArgumentTypeError("threshold must be 'auto' or a number between 0 and 1")
    return threshold


def train_candidate(
    name: str,
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    threshold_setting: str | float,
):
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            ("classifier", classifier),
        ]
    )
    model.fit(X_train, y_train)
    probs = denied_probability(model, X_val)

    if threshold_setting == "auto":
        threshold, metrics = select_threshold(y_val, probs, beta=2.0)
    else:
        threshold = float(threshold_setting)
        y_pred = pd.Series(probs).ge(threshold).map({True: "DENIED", False: "PAID"}).to_numpy()
        metrics = evaluate_predictions(y_val, y_pred, probs)
        metrics["threshold"] = threshold

    return name, model, threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cost-sensitive claim status models.")
    parser.add_argument("--data-dir", default=".", help="Directory containing source CSV files.")
    parser.add_argument(
        "--model-out",
        default="artifacts/best_model.joblib",
        help="Output path for selected model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/validation_metrics.json",
        help="Output path for validation metrics JSON.",
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument(
        "--threshold",
        type=parse_threshold,
        default="auto",
        help="DENIED probability threshold or 'auto' for validation F2 tuning.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y = load_and_prepare_data(args.data_dir)
    splits = split_data(X, y, random_state=args.random_state)

    candidates = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=args.random_state,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=args.random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=150,
                random_state=args.random_state,
            ),
        ),
    ]

    results = [
        train_candidate(
            name,
            classifier,
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            args.threshold,
        )
        for name, classifier in candidates
    ]

    best_name, best_model, best_threshold, best_metrics = sorted(
        results,
        key=lambda item: (
            item[3].get("average_precision", 0.0),
            item[3]["denied_f2"],
            item[3]["denied_recall"],
        ),
        reverse=True,
    )[0]

    artifact = {
        "model_name": best_name,
        "threshold": best_threshold,
        "random_state": args.random_state,
        "feature_columns": list(splits.X_train.columns),
        "model": best_model,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {"threshold": threshold, **metrics}
        for name, _, threshold, metrics in results
    }
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, _, threshold, metrics in results:
        rows.append(
            {
                "model": name,
                "threshold": threshold,
                "denied_precision": metrics["denied_precision"],
                "denied_recall": metrics["denied_recall"],
                "denied_f1": metrics["denied_f1"],
                "denied_f2": metrics["denied_f2"],
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
            }
        )
        (outputs_dir / f"validation_report_{name}.txt").write_text(
            metrics["classification_report"], encoding="utf-8"
        )

    pd.DataFrame(rows).sort_values(
        ["average_precision", "denied_f2"], ascending=False
    ).to_csv(outputs_dir / "validation_model_comparison.csv", index=False)

    summary = {
        "rows": int(len(X)),
        "features": list(X.columns),
        "best_model": best_name,
        "selected_threshold": best_threshold,
        "threshold_mode": args.threshold,
        "selection_metric": "validation average precision; DENIED F2 tie-break",
        "best_validation_metrics": {
            key: value
            for key, value in best_metrics.items()
            if key != "classification_report"
        },
    }
    (outputs_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved model artifact: {model_out}")


if __name__ == "__main__":
    main()
