from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SOURCE_FILES = ("ENCOUNTERS_DATA_PATH", "CLAIMS_DATA_PATH", "ORGANIZATIONS_DATA_PATH", "PATIENTS_DATA_PATH", "PROVIDERS_DATA_PATH")
TARGET_COLUMN = "claim_status"
DENIED_LABEL = "DENIED"
PAID_LABEL = "PAID"
# P0 safe-by-default reference profile. Only use fields that plausibly exist at
# intake; their actual availability must still be verified by a data owner.
# Do not add payment, settlement, future-outcome or cost fields without a
# timestamped feature-availability audit and independent review.
FEATURE_PROFILE = "pre_adjudication_minimal_v1"
DECISION_POINT = "After claim intake, before adjudication or claim-status changes"
FEATURE_COLUMNS = ["visit_type", "patient_age"]
EXCLUDED_FEATURE_COLUMNS = frozenset({
    "patient_gender", "total_visit_claim_cost", "claim_amount",
    "claim_status", "service_date", "claims_id", "patient_id",
})



@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _resolve_source_path(data_dir: Path, source_key: str) -> Path:
    value = os.environ.get(source_key)
    if not value:
        raise FileNotFoundError(
            f"Private source not configured: {source_key}. Set this environment variable."
        )
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_dir / path


def _read_csv(data_dir: Path, source_key: str) -> pd.DataFrame:
    path = _resolve_source_path(data_dir, source_key)
    if not path.is_file():
        raise FileNotFoundError(f"Configured source not found: {source_key}")
    return pd.read_csv(path)


def source_fingerprint(data_dir: str | Path) -> str:
    """Fingerprint exact input file bytes to prevent cross-dataset holdout evaluation.

    The digest is metadata only; it does not disclose claim/patient identifiers.
    """
    root = Path(data_dir)
    digest = hashlib.sha256()
    for source_key in SOURCE_FILES:
        path = _resolve_source_path(root, source_key)
        if not path.is_file():
            raise FileNotFoundError(f"Configured source not found: {source_key}")
        digest.update(source_key.encode("utf-8") + b"\\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _require_unique_nonnull(frame: pd.DataFrame, id_column: str, source: str) -> None:
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"{source}.{id_column} must contain unique, non-null identifiers.")


def load_and_prepare_data_with_metadata(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build model features with non-feature claim/patient/date provenance.

    Multiple claims may legitimately reference one encounter. Duplicate source
    identifiers, contradictory join keys, and orphan claim encounters fail closed.
    """
    root = Path(data_dir)
    encounters = _read_csv(root, "ENCOUNTERS_DATA_PATH").rename(columns={
        "ID": "visit_id", "PATIENT": "patient_id", "PROVIDER": "provider_id",
        "ORGANIZATION": "healthcare_organization_id", "ENCOUNTERCLASS": "visit_type",
        "TOTAL_CLAIM_COST": "total_visit_claim_cost",
    })
    claims = _read_csv(root, "CLAIMS_DATA_PATH").rename(columns={
        "ID": "claims_id", "ENCOUNTER": "visit_id", "PATIENT": "patient_id",
        "PROVIDER": "provider_id", "ORGANIZATION": "healthcare_organization_id",
        "SERVICE_DATE": "service_date", "CLAIM_AMOUNT": "claim_amount",
        "STATUS": TARGET_COLUMN,
    })
    organizations = _read_csv(root, "ORGANIZATIONS_DATA_PATH").rename(columns={
        "ID": "healthcare_organization_id", "NAME": "organization_name",
        "CITY": "organization_city", "STATE": "organization_state",
    })
    patients = _read_csv(root, "PATIENTS_DATA_PATH").rename(columns={
        "ID": "patients_id", "GENDER": "patient_gender",
        "BIRTHDATE": "patient_birthdate",
    })
    providers = _read_csv(root, "PROVIDERS_DATA_PATH").rename(columns={
        "ID": "providers_id", "SPECIALITY": "providers_speciality",
    })
    _require_columns(encounters, [
        "visit_id", "patient_id", "provider_id", "healthcare_organization_id",
        "visit_type",
    ], "ENCOUNTERS_DATA_PATH")
    _require_columns(claims, [
        "claims_id", "visit_id", "patient_id", "provider_id",
        "healthcare_organization_id", "service_date", TARGET_COLUMN,
    ], "CLAIMS_DATA_PATH")
    _require_columns(patients, ["patients_id", "patient_birthdate"], "PATIENTS_DATA_PATH")
    _require_columns(providers, ["providers_id", "providers_speciality"], "PROVIDERS_DATA_PATH")
    _require_columns(organizations, [
        "healthcare_organization_id", "organization_name", "organization_city",
    ], "ORGANIZATIONS_DATA_PATH")

    for frame, column, name in [
        (encounters, "visit_id", "ENCOUNTERS_DATA_PATH"),
        (claims, "claims_id", "CLAIMS_DATA_PATH"),
        (patients, "patients_id", "PATIENTS_DATA_PATH"),
        (providers, "providers_id", "PROVIDERS_DATA_PATH"),
        (organizations, "healthcare_organization_id", "ORGANIZATIONS_DATA_PATH"),
    ]:
        _require_unique_nonnull(frame, column, name)

    if claims["visit_id"].isna().any() or not claims["visit_id"].isin(encounters["visit_id"]).all():
        raise ValueError("CLAIMS_DATA_PATH contains a missing or unknown encounter ID.")

    claims_for_join = claims[[
        "claims_id", "visit_id", "patient_id", "provider_id",
        "healthcare_organization_id", "service_date", TARGET_COLUMN,
    ]].rename(columns={
        "patient_id": "claim_patient_id",
        "provider_id": "claim_provider_id",
        "healthcare_organization_id": "claim_organization_id",
    })
    encounter_fields = [
        "visit_id", "patient_id", "provider_id",
        "healthcare_organization_id", "visit_type",
    ]
    merged = encounters[encounter_fields].merge(
        claims_for_join, on="visit_id", how="inner", validate="one_to_many",
    )
    for encounter_key, claim_key in [
        ("patient_id", "claim_patient_id"),
        ("provider_id", "claim_provider_id"),
        ("healthcare_organization_id", "claim_organization_id"),
    ]:
        if (merged[encounter_key].isna() | merged[claim_key].isna()
                | merged[encounter_key].ne(merged[claim_key])).any():
            raise ValueError(f"Claim and encounter disagree on {encounter_key}.")

    merged = merged.merge(
        patients[["patients_id", "patient_birthdate"]],
        left_on="patient_id", right_on="patients_id", how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        organizations[["healthcare_organization_id", "organization_name", "organization_city"]],
        on="healthcare_organization_id", how="left", validate="many_to_one",
    )
    merged = merged.merge(
        providers[["providers_id", "providers_speciality"]],
        left_on="provider_id", right_on="providers_id", how="left",
        validate="many_to_one",
    )
    for column in ["patients_id", "organization_name", "providers_id"]:
        if merged[column].isna().any():
            raise ValueError(f"Unmatched reference record after joining {column}.")

    # Do not silently discard pending, reversed or unknown statuses: the owner
    # must explicitly extract mature historical labels before training.
    invalid_status = ~merged[TARGET_COLUMN].isin([PAID_LABEL, DENIED_LABEL])
    if invalid_status.any():
        raise ValueError(
            "Claims contain missing/non-final statuses. Supply a documented, "
            "approved mature PAID/DENIED cohort; do not silently filter labels."
        )
    merged = merged.copy()
    if merged.empty or merged[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Dataset must contain both PAID and DENIED claims.")
    merged["service_date"] = pd.to_datetime(merged["service_date"], errors="coerce")
    merged["patient_birthdate"] = pd.to_datetime(merged["patient_birthdate"], errors="coerce")
    if merged["service_date"].isna().any():
        raise ValueError("PAID/DENIED claims must have valid service dates.")
    age_days = (merged["service_date"] - merged["patient_birthdate"]).dt.days
    merged["patient_age"] = np.floor(age_days / 365.2425)
    merged.loc[(merged["patient_age"] < 0) | (merged["patient_age"] > 120), "patient_age"] = np.nan

    X = merged[FEATURE_COLUMNS].reset_index(drop=True)
    if set(X.columns) & EXCLUDED_FEATURE_COLUMNS:
        raise RuntimeError("Unapproved claim-cost, outcome or sensitive feature detected.")
    y = merged[TARGET_COLUMN].reset_index(drop=True)
    metadata = merged[["claims_id", "patient_id", "service_date"]].reset_index(drop=True)
    return X, y, metadata


def load_and_prepare_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Compatibility helper for feature inspection; training uses provenance too."""
    X, y, _ = load_and_prepare_data_with_metadata(data_dir)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    groups: pd.Series,
    random_state: int = 42,
) -> DatasetSplits:
    """Approximate 60/20/20 split with disjoint patient IDs across partitions.

    This is a patient-grouped reference split, NOT an out-of-time evaluation.
    Fail if the available groups cannot support both statuses in all three sets.
    """
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and patient groups must have equal length.")
    if groups.isna().any() or groups.nunique() < 5:
        raise ValueError("At least five non-null patient groups are required.")
    if set(y.unique()) != {PAID_LABEL, DENIED_LABEL}:
        raise ValueError("Only PAID and DENIED labels are supported.")
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    indices = np.arange(len(X))
    try:
        fold_indices = [test for _, test in folds.split(X, y, groups)]
    except ValueError as exc:
        raise ValueError("Not enough patient groups for grouped stratification.") from exc
    test_idx, val_idx = fold_indices[0], fold_indices[1]
    train_idx = np.setdiff1d(indices, np.concatenate((test_idx, val_idx)))
    for name, chosen in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        if set(y.iloc[chosen].unique()) != {PAID_LABEL, DENIED_LABEL}:
            raise ValueError(f"{name} split lacks both PAID and DENIED labels.")
    for left, right in [(train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)]:
        if set(groups.iloc[left]) & set(groups.iloc[right]):
            raise RuntimeError("Patient-group overlap detected between splits.")
    return DatasetSplits(
        X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx],
    )

def _normalize_categorical_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert StringDtype's pd.NA to np.nan for sklearn's SimpleImputer."""
    return frame.astype(object).where(pd.notna(frame), np.nan)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Pandas 3 can infer text as StringDtype instead of object. Classify
    # numeric columns explicitly, so strings never reach median imputation.
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("normalize_missing", FunctionTransformer(
                _normalize_categorical_missing_values, validate=False,
            )),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    return ColumnTransformer(transformers=transformers)


def denied_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    if DENIED_LABEL not in classes:
        raise ValueError(f"'{DENIED_LABEL}' label not present in classes: {classes}")
    return model.predict_proba(X)[:, classes.index(DENIED_LABEL)]


def threshold_predict(model: Pipeline, X: pd.DataFrame, threshold: float) -> np.ndarray:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    probs = denied_probability(model, X)
    return np.where(probs >= threshold, DENIED_LABEL, PAID_LABEL)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    denied_probs: np.ndarray | None = None,
) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[DENIED_LABEL],
        average=None,
        zero_division=0,
    )
    y_binary = (np.asarray(y_true) == DENIED_LABEL).astype(int)
    pred_binary = (np.asarray(y_pred) == DENIED_LABEL).astype(int)
    result = {
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
        "denied_precision": float(precision[0]),
        "denied_recall": float(recall[0]),
        "denied_f1": float(f1[0]),
        "denied_f2": float(fbeta_score(y_binary, pred_binary, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[DENIED_LABEL, PAID_LABEL]
        ).tolist(),
    }
    if denied_probs is not None:
        result["roc_auc"] = float(roc_auc_score(y_binary, denied_probs))
        result["average_precision"] = float(average_precision_score(y_binary, denied_probs))
    return result


def select_threshold(
    y_true: pd.Series,
    denied_probs: np.ndarray,
    *,
    beta: float = 2.0,
    minimum: float = 0.10,
    maximum: float = 0.90,
    step: float = 0.01,
) -> tuple[float, dict]:
    """Choose a validation threshold that maximizes DENIED-class F-beta."""
    if minimum <= 0 or maximum >= 1 or minimum >= maximum or step <= 0:
        raise ValueError("Invalid threshold search range.")

    y_binary = (np.asarray(y_true) == DENIED_LABEL).astype(int)
    best: tuple[float, float, float] | None = None
    best_metrics: dict | None = None

    for threshold in np.arange(minimum, maximum + 1e-9, step):
        y_pred = np.where(denied_probs >= threshold, DENIED_LABEL, PAID_LABEL)
        pred_binary = (y_pred == DENIED_LABEL).astype(int)
        fbeta = float(fbeta_score(y_binary, pred_binary, beta=beta, zero_division=0))
        recall = float(
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=[DENIED_LABEL],
                average=None,
                zero_division=0,
            )[1][0]
        )
        candidate = (fbeta, recall, -abs(float(threshold) - 0.5))
        if best is None or candidate > best:
            best = candidate
            best_metrics = evaluate_predictions(y_true, y_pred, denied_probs)
            best_metrics["threshold"] = float(round(threshold, 4))

    assert best_metrics is not None
    return float(best_metrics["threshold"]), best_metrics
