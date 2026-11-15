import numpy as np
import pandas as pd

from src.pipeline import (
    DENIED_LABEL,
    PAID_LABEL,
    build_preprocessor,
    load_and_prepare_data,
    select_threshold,
)


def test_preprocessor_handles_missing_values() -> None:
    X = pd.DataFrame(
        {
            "visit_type": ["ambulatory", None, "emergency"],
            "patient_gender": ["M", "F", None],
            "patient_age": [40.0, np.nan, 52.0],
            "total_visit_claim_cost": [100.0, 250.0, np.nan],
            "claim_amount": [80.0, np.nan, 300.0],
        }
    )
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == 3
    assert np.isfinite(transformed).all()



def test_preprocessor_handles_pandas_string_extension_dtype() -> None:
    X = pd.DataFrame({
        "visit_type": pd.Series(["ambulatory", pd.NA, "emergency"], dtype="string"),
        "patient_gender": pd.Series(["M", "F", pd.NA], dtype="string"),
        "patient_age": [40.0, np.nan, 52.0],
        "total_visit_claim_cost": [100.0, 250.0, np.nan],
        "claim_amount": [80.0, np.nan, 300.0],
    })
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()


def test_threshold_selection_prefers_valid_threshold() -> None:
    y = pd.Series([DENIED_LABEL, DENIED_LABEL, PAID_LABEL, PAID_LABEL])
    probs = np.array([0.9, 0.7, 0.4, 0.1])
    threshold, metrics = select_threshold(y, probs)
    assert 0 < threshold < 1
    assert metrics["denied_recall"] >= 0.5
    assert "denied_f2" in metrics


def test_patient_age_uses_service_date(tmp_path) -> None:
    pd.DataFrame(
        {
            "ID": ["v1", "v2"],
            "PATIENT": ["p1", "p1"],
            "PROVIDER": ["pr1", "pr1"],
            "ORGANIZATION": ["o1", "o1"],
            "ENCOUNTERCLASS": ["ambulatory", "emergency"],
            "TOTAL_CLAIM_COST": [100.0, 200.0],
        }
    ).to_csv(tmp_path / "encounters.csv", index=False)

    pd.DataFrame(
        {
            "ID": ["c1", "c2"],
            "ENCOUNTER": ["v1", "v2"],
            "PATIENT": ["p1", "p1"],
            "PROVIDER": ["pr1", "pr1"],
            "ORGANIZATION": ["o1", "o1"],
            "SERVICE_DATE": ["2020-06-01", "2021-06-01"],
            "CLAIM_AMOUNT": [80.0, 150.0],
            "STATUS": ["PAID", "DENIED"],
        }
    ).to_csv(tmp_path / "claims.csv", index=False)

    pd.DataFrame(
        {
            "ID": ["p1"],
            "NAME": ["Patient"],
            "GENDER": ["F"],
            "BIRTHDATE": ["1980-06-01"],
            "CITY": ["X"],
            "STATE": ["Y"],
        }
    ).to_csv(tmp_path / "patients.csv", index=False)

    pd.DataFrame(
        {"ID": ["o1"], "NAME": ["Org"], "CITY": ["X"], "STATE": ["Y"]}
    ).to_csv(tmp_path / "organizations.csv", index=False)

    pd.DataFrame(
        {
            "ID": ["pr1"],
            "NAME": ["Provider"],
            "SPECIALITY": ["General"],
            "GENDER": ["M"],
            "ORGANIZATION": ["o1"],
        }
    ).to_csv(tmp_path / "providers.csv", index=False)

    X, y = load_and_prepare_data(tmp_path)
    assert X["patient_age"].tolist() == [40.0, 41.0]
    assert set(y) == {"PAID", "DENIED"}
