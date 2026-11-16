"""Regression tests use entirely synthetic identifiers, never real patient records."""

from __future__ import annotations

import json
import joblib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    FEATURE_COLUMNS,
    FEATURE_PROFILE,
    DECISION_POINT,
    load_and_prepare_data,
    load_and_prepare_data_with_metadata,
    source_fingerprint,
    split_data,
)


@pytest.fixture
def synthetic_claim_files(tmp_path):
    patients = [
        {"ID": f"p{i}", "GENDER": "F" if i % 2 else "M",
         "BIRTHDATE": "1980-01-01"}
        for i in range(30)
    ]
    encounters = []
    claims = []
    for i in range(30):
        for j in range(2):
            visit_id = f"v{i}-{j}"
            encounters.append({
                "ID": visit_id, "PATIENT": f"p{i}", "PROVIDER": "provider-1",
                "ORGANIZATION": "org-1",
                "ENCOUNTERCLASS": "ambulatory" if j == 0 else "emergency",
                "TOTAL_CLAIM_COST": 100.0 + 3 * i + j,
            })
            claims.append({
                "ID": f"c{i}-{j}", "ENCOUNTER": visit_id,
                "PATIENT": f"p{i}", "PROVIDER": "provider-1",
                "ORGANIZATION": "org-1",
                "SERVICE_DATE": "2020-01-20", "CLAIM_AMOUNT": 50 + 2 * i + j,
                "STATUS": "PAID" if j == 0 else "DENIED",
            })
    pd.DataFrame(encounters).to_csv(tmp_path / "encounters.csv", index=False)
    pd.DataFrame(claims).to_csv(tmp_path / "claims.csv", index=False)
    pd.DataFrame(patients).to_csv(tmp_path / "patients.csv", index=False)
    pd.DataFrame([{"ID": "provider-1", "SPECIALITY": "GENERAL PRACTICE"}]).to_csv(
        tmp_path / "providers.csv", index=False
    )
    pd.DataFrame([{
        "ID": "org-1", "NAME": "Training Organization",
        "CITY": "Example City",
    }]).to_csv(tmp_path / "organizations.csv", index=False)
    return tmp_path


def test_patient_groups_are_disjoint(synthetic_claim_files):
    X, y, metadata = load_and_prepare_data_with_metadata(synthetic_claim_files)
    split = split_data(X, y, groups=metadata["patient_id"])
    groups = [
        set(metadata.loc[part.index, "patient_id"])
        for part in [split.X_train, split.X_val, split.X_test]
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sum(map(len, [split.X_train, split.X_val, split.X_test])) == len(X)
    assert all(set(labels) == {"PAID", "DENIED"} for labels in (
        split.y_train, split.y_val, split.y_test
    ))


def test_group_split_rejects_insufficient_groups():
    X = pd.DataFrame({"patient_age": [20., 30., 40., 50.]})
    y = pd.Series(["PAID", "DENIED", "PAID", "DENIED"])
    with pytest.raises(ValueError, match="five"):
        split_data(X, y, groups=pd.Series(["p1", "p2", "p3", "p4"]))


def test_duplicate_encounter_identifier_fails_closed(synthetic_claim_files):
    path = synthetic_claim_files / "encounters.csv"
    df = pd.read_csv(path)
    df.loc[1, "ID"] = df.loc[0, "ID"]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_and_prepare_data(synthetic_claim_files)


def test_orphan_claim_fails_closed(synthetic_claim_files):
    path = synthetic_claim_files / "claims.csv"
    df = pd.read_csv(path)
    df.loc[0, "ENCOUNTER"] = "absent-visit"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown encounter"):
        load_and_prepare_data(synthetic_claim_files)


def test_claim_encounter_patient_mismatch_fails_closed(synthetic_claim_files):
    path = synthetic_claim_files / "claims.csv"
    df = pd.read_csv(path)
    df.loc[0, "PATIENT"] = "p1"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree on patient_id"):
        load_and_prepare_data(synthetic_claim_files)


def test_missing_dimension_record_fails_closed(synthetic_claim_files):
    path = synthetic_claim_files / "patients.csv"
    df = pd.read_csv(path)
    df = df.loc[df["ID"] != "p0"]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unmatched reference"):
        load_and_prepare_data(synthetic_claim_files)


def test_multiple_claims_per_encounter_do_not_multiply_rows(synthetic_claim_files):
    path = synthetic_claim_files / "claims.csv"
    df = pd.read_csv(path)
    extra = df.iloc[[0]].copy()
    extra["ID"] = "extra-claim"
    df = pd.concat((df, extra), ignore_index=True)
    df.to_csv(path, index=False)
    X, _, metadata = load_and_prepare_data_with_metadata(synthetic_claim_files)
    assert len(X) == len(df) == len(metadata)
    assert metadata["claims_id"].nunique() == len(metadata)


def test_dataset_fingerprint_detects_changes(synthetic_claim_files):
    original = source_fingerprint(synthetic_claim_files)
    assert len(original) == 64
    assert source_fingerprint(synthetic_claim_files) == original
    path = synthetic_claim_files / "claims.csv"
    df = pd.read_csv(path)
    df.loc[0, "CLAIM_AMOUNT"] = 999
    df.to_csv(path, index=False)
    assert source_fingerprint(synthetic_claim_files) != original


def test_train_evaluate_and_reject_changed_data(synthetic_claim_files, tmp_path):
    model = tmp_path / "artifacts" / "model.joblib"
    metrics = tmp_path / "artifacts" / "test_metrics.json"
    outputs = tmp_path / "output"
    train = [
        sys.executable, "-m", "src.train", "--data-dir", str(synthetic_claim_files),
        "--model-out", str(model), "--outputs-dir", str(outputs),
        "--metrics-out", str(tmp_path / "artifacts" / "validation.json"),
    ]
    subprocess.run(train, check=True, capture_output=True, text=True)
    assert model.exists()
    evaluate = [
        sys.executable, "-m", "src.evaluate", "--data-dir", str(synthetic_claim_files),
        "--model-path", str(model), "--outputs-dir", str(outputs),
        "--metrics-out", str(metrics),
    ]
    subprocess.run(evaluate, check=True, capture_output=True, text=True)
    payload = json.loads(metrics.read_text())
    assert payload["split_method"] == "stratified_patient_group_5_fold"
    assert payload["feature_profile"] == FEATURE_PROFILE
    assert payload["evidence_status"] == "unverified_local_holdout_research_only"
    assert payload["independently_reproduced"] is False
    assert payload["out_of_time_validated"] is False
    assert payload["deployment_approved"] is False
    bundle = joblib.load(model)
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert bundle["feature_profile"] == FEATURE_PROFILE
    assert bundle["decision_point"] == DECISION_POINT
    assert payload["source_sha256"] == source_fingerprint(synthetic_claim_files)
    assert "baseline_all_denied" in payload
    assert "baseline_all_paid" in payload
    assert len(pd.read_csv(outputs / "test_predictions.csv")) == payload["split_rows"]["test"]
    data = pd.read_csv(synthetic_claim_files / "claims.csv")
    data.loc[0, "CLAIM_AMOUNT"] += 1
    data.to_csv(synthetic_claim_files / "claims.csv", index=False)
    failed = subprocess.run(evaluate, check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "Source CSV files differ" in failed.stderr


def test_intake_features_ignore_excluded_cost_and_gender_values(synthetic_claim_files):
    original, _, _ = load_and_prepare_data_with_metadata(synthetic_claim_files)
    assert list(original.columns) == FEATURE_COLUMNS
    assert "claim_amount" not in original.columns
    assert "total_visit_claim_cost" not in original.columns
    assert "patient_gender" not in original.columns

    changes = [
        ("claims.csv", "CLAIM_AMOUNT", 999999.0),
        ("encounters.csv", "TOTAL_CLAIM_COST", 9999999.0),
        ("patients.csv", "GENDER", "X"),
    ]
    for filename, col, replacement in changes:
        path = synthetic_claim_files / filename
        df = pd.read_csv(path)
        df[col] = replacement
        df.to_csv(path, index=False)

    changed, _, _ = load_and_prepare_data_with_metadata(synthetic_claim_files)
    pd.testing.assert_frame_equal(original, changed)


def test_missing_unapproved_cost_gender_columns_are_accepted(synthetic_claim_files):
    for filename, col in [
        ("claims.csv", "CLAIM_AMOUNT"),
        ("encounters.csv", "TOTAL_CLAIM_COST"),
        ("patients.csv", "GENDER"),
    ]:
        path = synthetic_claim_files / filename
        df = pd.read_csv(path).drop(columns=[col])
        df.to_csv(path, index=False)

    X, y, _ = load_and_prepare_data_with_metadata(synthetic_claim_files)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y) == {"PAID", "DENIED"}


def test_unknown_or_pending_claim_label_fails_closed(synthetic_claim_files):
    path = synthetic_claim_files / "claims.csv"
    df = pd.read_csv(path)
    df.loc[0, "STATUS"] = "PENDING"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing/non-final statuses"):
        load_and_prepare_data_with_metadata(synthetic_claim_files)
