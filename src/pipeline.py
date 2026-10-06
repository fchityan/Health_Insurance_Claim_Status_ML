from __future__ import annotations

from dataclasses import dataclass
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "claim_status"
DENIED_LABEL = "DENIED"
PAID_LABEL = "PAID"
FEATURE_COLUMNS = [
    "visit_type",
    "patient_gender",
    "patient_age",
    "total_visit_claim_cost",
    "claim_amount",
]


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def load_and_prepare_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load source CSV files and prepare deterministic model-ready features."""
    root = Path(data_dir)

    encounters = _read_csv(root, "encounters.csv")
    claims = _read_csv(root, "claims.csv")
    organizations = _read_csv(root, "organizations.csv")
    patients = _read_csv(root, "patients.csv")
    providers = _read_csv(root, "providers.csv")

    encounters = encounters.rename(
        columns={
            "ID": "visit_id",
            "PATIENT": "patient_id",
            "PROVIDER": "provider_id",
            "ORGANIZATION": "healthcare_organization_id",
            "ENCOUNTERCLASS": "visit_type",
            "TOTAL_CLAIM_COST": "total_visit_claim_cost",
        }
    )
    claims = claims.rename(
        columns={
            "ID": "claims_id",
            "ENCOUNTER": "visit_id",
            "PATIENT": "patient_id",
            "PROVIDER": "provider_id",
            "ORGANIZATION": "healthcare_organization_id",
            "SERVICE_DATE": "service_date",
            "CLAIM_AMOUNT": "claim_amount",
            "STATUS": TARGET_COLUMN,
        }
    )
    organizations = organizations.rename(
        columns={
            "ID": "healthcare_organization_id",
            "NAME": "organization_name",
            "CITY": "organization_city",
            "STATE": "organization_state",
        }
    )
    patients = patients.rename(
        columns={
            "ID": "patients_id",
            "NAME": "patient_name",
            "GENDER": "patient_gender",
            "BIRTHDATE": "patient_birthdate",
            "CITY": "patient_city",
            "STATE": "patient_state",
        }
    )
    providers = providers.rename(
        columns={
            "ID": "providers_id",
            "NAME": "providers_name",
            "SPECIALITY": "providers_speciality",
            "GENDER": "providers_gender",
            "ORGANIZATION": "healthcare_organization_id",
        }
    )

    merged = encounters.merge(
        claims[["visit_id", "service_date", "claim_amount", TARGET_COLUMN]],
        on="visit_id",
        how="inner",
    )
    merged = merged.merge(
        patients[["patients_id", "patient_gender", "patient_birthdate"]],
        left_on="patient_id",
        right_on="patients_id",
        how="left",
    )
    merged = merged.merge(
        organizations[["healthcare_organization_id", "organization_name", "organization_city"]],
        on="healthcare_organization_id",
        how="left",
    )
    merged = merged.merge(
        providers[["providers_id", "providers_speciality"]],
        left_on="provider_id",
        right_on="providers_id",
        how="left",
    )

    merged["patient_birthdate"] = pd.to_datetime(merged["patient_birthdate"], errors="coerce")
    merged["service_date"] = pd.to_datetime(merged["service_date"], errors="coerce")
    age_days = (merged["service_date"] - merged["patient_birthdate"]).dt.days
    merged["patient_age"] = np.floor(age_days / 365.2425)
    merged.loc[merged["patient_age"] < 0, "patient_age"] = np.nan

    merged = merged[merged[TARGET_COLUMN].isin([PAID_LABEL, DENIED_LABEL])].copy()

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after merge: {missing_cols}")

    model_df = merged[required].copy()
    model_df = model_df.dropna(subset=[TARGET_COLUMN])
    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]
    if y.nunique() < 2:
        raise ValueError("Dataset must contain both PAID and DENIED claims.")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> DatasetSplits:
    """Create 60/20/20 train/validation/test split with stratification."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
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
