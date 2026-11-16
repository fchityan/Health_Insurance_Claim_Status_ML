# Health Insurance Claims — Historical Denial-Risk Screening for Human Review

**Portfolio research prototype · historical claim-status classification · not automated claims adjudication**

This project explores whether limited information available around claim intake could help an insurance examiner **prioritise some submissions for earlier human review**. Its supervised target is a historically recorded `PAID` or `DENIED` **status**, *not* medical necessity, fraud, coverage entitlement, the correctness of an earlier decision, or a recommendation to deny a claim.

**Validation status: NOT VERIFIED ON AUTHORISED SOURCE DATA.** Source CSVs are not in this repository, updated holdout accuracy is not independently reproduced, and no production or insurer approval is implied. The original notebook is historical academic exploration; its statistical tests about provider costs address a *different* question and do **not** establish fraud.

## Intended decision and explicit boundary

| Question | Research-prototype answer |
|---|---|
| **Who might use it?** | A claims examiner or operations lead overseeing an ordinary manual review queue |
| **When could a score be calculated?** | **After claim intake and encounter context are available, before any adjudication, settlement or status update** (proposed; no intake timestamps are present to verify this in the source) |
| **What is predicted?** | The probability of the **historically recorded `DENIED` outcome** under the available labels, not whether the claim *ought* to be denied |
| **What could the output influence?** | *Review priority only*, after an insurer has approved the use case, costs, fairness controls and examiner capacity |
| **Who makes the final decision?** | An authorised human examiner following the policy and established claims process |
| **What is prohibited?** | Using the score to automatically refuse benefits, assert fraud, infer policy eligibility or represent actual operational improvement |

**Research success criteria to validate, not achieved results:** Compare against ordinary queue ordering and always-PAID/always-DENIED baselines; report precision/recall of historically denied claims at a *fixed examiner review capacity*, the number of false escalations and, where valid, workload or avoidable rework. Objectives, cutoff, cost weights, label maturity and acceptance targets require claims-owner approval.

## P0 safety change: conservative feature profile

The maintained ML pipeline now uses a single **`pre_adjudication_minimal_v1`** feature profile:

| Model input | Proposed origin | Evidence gap |
|---|---|---|
| `visit_type` | Encounter classification | Prove it is populated **at or before** claim intake |
| `patient_age` | Derived from birthdate and service date | Prove both dates are permitted and available at the defined cutoff |

The **default pipeline does not train on** `claim_amount`, `total_visit_claim_cost`, `patient_gender`, patient/claim identifiers or historical claim status (except the status as training target). Cost fields could be calculated or amended after the decision and are **unapproved for this stage**; gender is omitted to minimise sensitive/proxy features. These exclusions reduce known leakage risks but **do not prove as-of-time validity**. A data owner must confirm the feature timestamps and intended operational point before any claims-use conclusion.

Unknown, missing, pending or non-final claim-status labels now cause training to **fail closed** rather than silently excluding those records. The data owner must supply an authorised cohort with an agreed final-status observation window, including a documented rule for reversals and appeals.

Older model artifacts using the previous feature set are deliberately rejected by the evaluator and **must be retrained**.

## Implemented technical evidence

- Multi-table referential checks for encounters, claims, patients, providers and organisations; inconsistent IDs and ambiguous joins fail closed.
- Stable age-at-service engineering, safe handling of missing features and categorical encoding in scikit-learn pipelines.
- Patient-separated stratified train/validation/test folds (approximately 60%/20%/20%); these **are not** an out-of-time validation.
- Logistic Regression, Random Forest and Gradient Boosting, compared on validation average precision with DENIED F2 as a tie-break.
- Validation-only threshold selection, a locked test evaluation path, source SHA-256 fingerprint and strict feature-profile checks.
- Always-PAID and always-DENIED comparisons so predicting every case as DENIED cannot be mistaken for useful recall.
- Synthetic-only regression tests for missing features, unsafe claim labels, join errors, group leakage, legacy model protection and reproducible evaluation.

The selection threshold remains a **research heuristic**, not an approved examiner workload rule. The test report is stamped `unverified_local_holdout_research_only`, `independently_reproduced: false`, `out_of_time_validated: false` and `deployment_approved: false` even when numerical metrics are generated locally.

## Reproduce safely

Use only datasets you are authorised to process. The required local CSV files are `encounters.csv`, `claims.csv`, `patients.csv`, `providers.csv`, and `organizations.csv`. The source CSVs are **not committed**, nor are model artifacts or outputs.

```bash
python -m pip install -r requirements.txt
python -m pytest -q

# Train three models on a locally authorised mature-label cohort
python -m src.train --data-dir . --model-out artifacts/best_model.joblib \
  --outputs-dir outputs --threshold auto

# Evaluate the locked grouped holdout, using the exact same source file bytes
python -m src.evaluate --data-dir . --model-path artifacts/best_model.joblib \
  --outputs-dir outputs
```

Outputs include validation comparisons, a locked-test classification report, DENIED precision/recall/F1/F2, ROC-AUC, average precision, a confusion matrix, two trivial baselines, source fingerprints and an explicitly **unverified** evidence label. Do not publish row-level data, source files or generated metrics from unauthorised/untested datasets. `joblib` loading can execute code: evaluate trusted local models only.

The source data are not available in this repository; **no independent performance claim can currently be made**. If authorised data later become available, run the locked evaluation, repeat it independently, verify as-of feature availability, evaluate out-of-time generalisation and report the results *with denominator, date window, uncertainty and baseline*.

## Project map

| Purpose | Evidence |
|---|---|
| Current feature/join/split implementation | [Data pipeline](src/pipeline.py) |
| Model training and threshold selection | [Training workflow](src/train.py) |
| Locked evaluation and evidence flags | [Evaluation workflow](src/evaluate.py) |
| Synthetic regression tests | [Pipeline tests](tests/test_pipeline.py) · [Integrity tests](tests/test_integrity.py) |
| Business requirement and decision boundary | [Claims business requirements](docs/enterprise/business-requirements-and-traceability.md) |
| Model assurance and outstanding risks | [Validation gap register](docs/enterprise/validation-gap-register.md) |
| How to establish external evidence | [P0 evidence and data gates](docs/enterprise/p0-evidence-gates.md) |
| Architecture and governance | [Enterprise documentation](docs/enterprise/README.md) |
| Original, separate exploratory provider-cost research | [Historical academic notebook](Health_Insurance_Claim_Status_ML.ipynb) |

### Release position

The code implements safer defaults and reproducibility controls, but **none of the business, external-data, temporal-validation, examiner acceptance or privacy gates are approved**. The application is a technical/reference project only; not a real-time claim-decision or claims-review service.
