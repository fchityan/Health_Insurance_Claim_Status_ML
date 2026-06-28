from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    """Load source CSV files and prepare model-ready features/target."""
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
        claims[["visit_id", "claim_amount", TARGET_COLUMN]],
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
    current_year = pd.Timestamp.today().year
    merged["patient_age"] = current_year - merged["patient_birthdate"].dt.year

    merged = merged[merged[TARGET_COLUMN].isin([PAID_LABEL, DENIED_LABEL])].copy()

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after merge: {missing_cols}")

    model_df = merged[required].dropna().copy()
    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]
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

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )


def denied_probability(model: Pipeline, X: pd.DataFrame, denied_label: str = DENIED_LABEL) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    if denied_label not in classes:
        raise ValueError(f"'{denied_label}' label not present in classes: {classes}")
    idx = classes.index(denied_label)
    return model.predict_proba(X)[:, idx]


def threshold_predict(
    model: Pipeline,
    X: pd.DataFrame,
    threshold: float,
    denied_label: str = DENIED_LABEL,
    paid_label: str = PAID_LABEL,
) -> np.ndarray:
    probs = denied_probability(model, X, denied_label=denied_label)
    return np.where(probs >= threshold, denied_label, paid_label)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[DENIED_LABEL],
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[DENIED_LABEL, PAID_LABEL])
    return {
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "denied_precision": float(precision[0]),
        "denied_recall": float(recall[0]),
        "denied_f1": float(f1[0]),
        "confusion_matrix": cm.tolist(),
    }
