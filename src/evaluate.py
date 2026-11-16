from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import (
    FEATURE_COLUMNS,
    FEATURE_PROFILE,
    DECISION_POINT,
    denied_probability,
    evaluate_predictions,
    load_and_prepare_data_with_metadata,
    source_fingerprint,
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

    # Only load model bundles from a trusted source: joblib deserialization can
    # execute arbitrary code. Do not evaluate model artifacts from strangers.
    artifact = joblib.load(args.model_path)
    expected_digest = artifact.get("source_sha256")
    if not expected_digest:
        raise ValueError("Legacy model artifact lacks a source fingerprint; retrain first.")
    current_digest = source_fingerprint(args.data_dir)
    if current_digest != expected_digest:
        raise ValueError(
            "Source CSV files differ from training. Refusing to reuse a locked-test split."
        )
    if artifact.get("split_method") != "stratified_patient_group_5_fold":
        raise ValueError("Model artifact lacks a supported patient-grouped split; retrain first.")

    if artifact.get("feature_profile") != FEATURE_PROFILE or artifact.get("decision_point") != DECISION_POINT:
        raise ValueError("Artifact uses an unapproved legacy feature profile; retrain before evaluation.")
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Artifact feature schema contains unapproved or missing intake fields.")

    model = artifact["model"]
    threshold = float(artifact["threshold"])
    random_state = int(artifact.get("random_state", 42))
    X, y, metadata = load_and_prepare_data_with_metadata(args.data_dir)
    splits = split_data(
        X, y, groups=metadata["patient_id"], random_state=random_state,
    )
    if list(splits.X_test.columns) != artifact.get("feature_columns"):
        raise ValueError("Model feature schema changed since training.")

    denied_probs = denied_probability(model, splits.X_test)
    y_pred = threshold_predict(model, splits.X_test, threshold)
    metrics = evaluate_predictions(splits.y_test, y_pred, denied_probs)
    metrics["threshold"] = threshold
    metrics["model_name"] = artifact.get("model_name", "unknown")
    metrics["feature_profile"] = FEATURE_PROFILE
    metrics["decision_point"] = DECISION_POINT
    metrics["evidence_status"] = "unverified_local_holdout_research_only"
    metrics["independently_reproduced"] = False
    metrics["out_of_time_validated"] = False
    metrics["deployment_approved"] = False
    metrics["limitations"] = (
        "Historical PAID/DENIED statuses are not coverage ground truth; "
        "source-data provenance, label maturity, feature timestamps, "
        "independent reproduction and temporal generalisation require approval."
    )
    metrics["split_method"] = artifact["split_method"]
    metrics["source_sha256"] = current_digest
    metrics["split_rows"] = {
        "train": int(len(splits.X_train)),
        "validation": int(len(splits.X_val)),
        "test": int(len(splits.X_test)),
    }
    # Compare to naive baselines so recall alone is not presented as meaningful
    # value: predicting DENIED for every claim always has DENIED recall of 1.
    for baseline_name, label in [
        ("all_paid", "PAID"), ("all_denied", "DENIED"),
    ]:
        baseline = evaluate_predictions(
            splits.y_test, np.full(len(splits.y_test), label, dtype=object),
        )
        metrics[f"baseline_{baseline_name}"] = {
            "denied_precision": baseline["denied_precision"],
            "denied_recall": baseline["denied_recall"],
            "denied_f2": baseline["denied_f2"],
            "confusion_matrix": baseline["confusion_matrix"],
        }

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
