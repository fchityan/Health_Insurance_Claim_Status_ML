# Health Insurance Claims — Historical Denial-Risk Screening for Human Review

**Insurance Analytics · Machine Learning · Claims Operations · Decision Support**

## Project overview

This project investigates whether historical health insurance claims data can support **earlier human review of submissions that may be associated with a recorded denial outcome**. It combines insurance operations knowledge with reproducible machine learning, data-quality controls and model-evaluation practices.

The intended application is **claims-review prioritisation**, not automated claim adjudication. The model learns from historical `PAID` and `DENIED` statuses; those records do **not** establish whether a claim was correctly paid or denied, whether coverage applies or whether fraud occurred.

| Business consideration | Proposed approach |
|---|---|
| **Business challenge** | Claims teams need consistent ways to identify cases that may warrant earlier examination |
| **Intended users** | Claims examiners and operations managers |
| **Decision point** | After claim intake and availability of encounter context, before adjudication or settlement |
| **Analytical output** | An estimated probability associated with a historically recorded denial status |
| **Human oversight** | An authorised examiner retains full responsibility for reviewing and deciding claims |
| **Potential value to evaluate** | More relevant cases identified at a fixed review capacity, without disproportionate false escalations |

The decision point is a proposed design, **not yet verified against the timestamps in an authorised dataset**. No real insurer has approved or deployed this workflow.

## Solution approach

```text
Historical claims and reference datasets
        ↓
Data-quality checks and record reconciliation
        ↓
Limited intake-stage feature engineering
        ↓
Patient-separated training and validation
        ↓
Model comparison and threshold selection
        ↓
Locked holdout evaluation and baseline comparison
        ↓
Research outputs for potential human-review prioritisation
```

The maintained classification pipeline joins encounters, claims, patients, providers and organisations with strict identifier and relationship checks. It handles missing values using fitted preprocessing pipelines and compares **Logistic Regression, Random Forest and Gradient Boosting**.

### Feature selection and safeguards

The current reference model uses only the following proposed intake-stage predictors:

| Feature | Description |
|---|---|
| `visit_type` | Encounter classification |
| `patient_age` | Age derived from birthdate and service date |

Claim amount, total visit cost and patient gender are excluded from the default feature set. Claim status is the **historical training label only**, never an input feature. Missing, pending or unexpected status values are rejected rather than silently discarded.

These controls reduce potential leakage and unnecessary feature exposure, but **do not independently confirm that the two inputs were available at the exact historical prediction time**.

## Model development and evaluation

- **Data preparation:** Referential integrity checks, feature preprocessing and deterministic age-at-service calculation.
- **Data splitting:** Patient-separated train, validation and test partitions (approximately 60% / 20% / 20%). This is not a future-period holdout.
- **Model comparison:** Three classification algorithms, selected on validation average precision with DENIED-class F2 as a tie-breaker.
- **Threshold selection:** Validation-only tuning of an illustrative recall-weighted threshold, separate from the locked test set.
- **Evaluation outputs:** Confusion matrix, DENIED precision/recall/F1/F2, ROC-AUC, average precision and always-PAID/always-DENIED baselines.
- **Reproducibility:** Source-file fingerprints, versioned model feature contract and automated tests using synthetic records.

**Validation status:** Automated tests verify software behaviour; they do **not** establish real-world prediction quality. The original authorised CSV inputs are not published, and results for the updated pipeline have not been independently reproduced on those data. A genuine operational assessment still requires approved outcome definitions, feature-timing checks, out-of-time validation, examiner-capacity analysis and privacy review.

## Solution components and documentation

| Area | Deliverable | Documentation / source |
|---|---|---|
| **Business analysis** | Proposed claims-review use case, stakeholders, requirements and decision rights | [Business requirements and traceability](docs/enterprise/business-requirements-and-traceability.md) |
| **Data engineering** | Multi-source assembly, integrity checks and limited feature preparation | [Data processing pipeline](src/pipeline.py) |
| **Predictive modelling** | Model training, comparison and research threshold selection | [Model training workflow](src/train.py) |
| **Model evaluation** | Locked holdout reporting, baseline comparison and provenance checks | [Evaluation workflow](src/evaluate.py) |
| **Quality assurance** | Synthetic regression tests for data joins, patient separation, label handling and feature safety | [Pipeline tests](tests/test_pipeline.py) · [Data-integrity tests](tests/test_integrity.py) |
| **Solution architecture** | Reference design, data contracts and technical decision record | [Architecture and data design](docs/enterprise/solution-architecture-and-data-contracts.md) |
| **Model governance** | Validation requirements, limitations, privacy considerations and operational controls | [Model assurance and operations](docs/enterprise/assurance-security-and-operations.md) · [Validation register](docs/enterprise/validation-gap-register.md) |
| **Exploratory research** | Original academic analysis of provider costs, separate from the current status-prediction workflow | [Historical analysis notebook](Health_Insurance_Claim_Status_ML.ipynb) |

## Reproducing the research workflow

The model expects five authorised local source CSVs for encounters, claims, patients, providers and organisations. Their physical filenames are not documented publicly. Set the `ENCOUNTERS_DATA_PATH`, `CLAIMS_DATA_PATH`, `ORGANIZATIONS_DATA_PATH`, `PATIENTS_DATA_PATH` and `PROVIDERS_DATA_PATH` environment variables to their local paths before training or evaluation. The source files and generated artefacts are not included in this public repository.

```bash
python -m pip install -r requirements.txt
python -m pytest -q

python -m src.train --data-dir . \
  --model-out artifacts/best_model.joblib \
  --outputs-dir outputs --threshold auto

python -m src.evaluate --data-dir . \
  --model-path artifacts/best_model.joblib \
  --outputs-dir outputs
```

Only use data you are authorised to process and trusted local model artefacts. The evaluation output is explicitly marked as unverified research evidence until independent replication and approval. Older model artefacts using the prior feature profile must be retrained.

## Project scope and limitations

This is a **research and portfolio implementation**, not a deployed claims-processing service. It does not automate payment or denial decisions, establish fraud or coverage eligibility, or demonstrate measured operational savings. Real-world use would require independently validated performance on appropriately mature claims outcomes, confirmed prediction-time feature availability, legal and privacy review, and human-examiner acceptance.
