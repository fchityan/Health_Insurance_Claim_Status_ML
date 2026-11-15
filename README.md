# Health Insurance Claim Status — Claims Analytics, ML Solution Design & Model Assurance

A reproducible study of **historically recorded** `PAID` versus `DENIED` claim statuses, using model comparison, patient-separated evaluation and threshold analysis. A predicted status is **not a determination that a claim should be paid or denied**.

The original notebook, `Health_Insurance_Claim_Status_ML.ipynb`, is preserved as historical academic analysis. Its provider-cost hypothesis tests address a **separate exploratory billing-risk question** and do not establish provider fraud. The maintained scripts are a predeployment modelling reference, not an automated claims-adjudication service.

## Intended Use and Decision Boundary

The model can be studied as a **potential claim-review prioritisation aid** under human oversight. `PAID` and `DENIED` are historical outcome labels, not verified coverage-policy ground truth. Before any pilot, an insurer must define the prediction cutoff, business objective, legal/privacy basis, adjudication policy, label provenance, review capacity and true cost of errors.

**No real insurer deployment or updated source-data performance is verified.** A high DENIED recall can be achieved merely by flagging every claim as DENIED; the evaluation now includes both always-PAID and always-DENIED baselines. Neither model predictions nor notebook findings justify withholding benefits.

## Business Analysis, Solution Design & Model Assurance

This project presents a reproducible approach to identifying claim-denial risk for **human review and prioritisation**. The material documents a modelling reference, not an automated claim payment decision or a production insurance deployment.

| Delivery Area | Supporting Evidence |
|---|---|
| Claims-status exploratory analysis | [Original analysis notebook](Health_Insurance_Claim_Status_ML.ipynb) |
| Data preparation, feature engineering & input controls | [Classification pipeline](src/pipeline.py) |
| Model comparison & validation-only threshold selection | [Training workflow](src/train.py) |
| Holdout evaluation & reporting workflow | [Evaluation script](src/evaluate.py) |
| Automated quality checks | [Unit tests](tests/test_pipeline.py) |

The source CSV datasets are not committed, and no unverified numerical model-performance claim is made.

## Project Documentation

[Health Insurance Claims Analytics](docs/enterprise/README.md). These are proposed enterprise artefacts grounded in the existing prototype, not evidence of live claims decisions or regulatory approval.

## What the project demonstrates

- Multi-table feature assembly from encounters, claims, patients, providers, and organizations
- Deterministic patient-age engineering using service date rather than the current calendar year
- Missing-value handling through model pipelines instead of dropping incomplete rows
- Numeric median imputation and categorical most-frequent imputation
- One-hot encoding for categorical variables
- Five-fold patient-grouped stratification (approximately 60/20/20), preventing any patient from appearing in multiple partitions
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Validation-only threshold tuning for the `DENIED` class
- Recall-weighted F2 scoring for cost-sensitive classification
- Probability-level evaluation using ROC-AUC and average precision
- Saved model artifacts and reproducible holdout evaluation
- Unit tests for threshold selection, preprocessing, join integrity, patient-separated splits, and reproducible evaluation

## Repository structure

```text
.
├── Health_Insurance_Claim_Status_ML.ipynb
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    └── test_pipeline.py
```

## Expected source files

Place these CSV files in one directory:

- `encounters.csv`
- `claims.csv`
- `organizations.csv`
- `patients.csv`
- `providers.csv`

The data files are not committed or included in published metrics. The five source filenames and the top-level `data/` directory are ignored by Git; do not commit raw healthcare or claim records anywhere else in this repository. Only use data for which you have authorization.

The loader requires unique, non-null claim/encounter/dimension IDs, matching encounter–claim patient/provider/organization IDs, and resolvable reference records. **It deliberately fails closed** on ambiguous joins or invalid service dates rather than silently changing the sample.

## Train models

```bash
pip install -r requirements.txt
python -m src.train \
  --data-dir . \
  --model-out artifacts/best_model.joblib \
  --outputs-dir outputs \
  --threshold auto
```

`--threshold auto` searches the validation set for a threshold that maximizes DENIED-class F2. You can also pass a fixed probability threshold such as `--threshold 0.40`.

Training uses five disjoint patient-group folds, allocating three to training, one to validation and one to test. Exact row proportions vary with group sizes and the workflow fails when a split cannot contain both labels. This is **not a time-based holdout**; future work must independently validate out-of-time generalization and whether `claim_amount` or `total_visit_claim_cost` were available before a real prediction cutoff.

Training records the source-file SHA-256 fingerprint, the grouped-split method and observed row counts in its model artefact and summary. Old pre-upgrade model files must be **retrained** before using the evaluator.

Training writes model-comparison outputs such as:

```text
artifacts/
├── best_model.joblib
└── validation_metrics.json

outputs/
├── training_summary.json
├── validation_model_comparison.csv
└── validation_report_<model>.txt
```

## Evaluate the locked model

```bash
python -m src.evaluate \
  --data-dir . \
  --model-path artifacts/best_model.joblib \
  --outputs-dir outputs
```

This reconstructs the **same patient-grouped holdout**, refuses evaluation if any of the five source CSV files differ from the training fingerprint, and requires a matching feature schema. Model files use `joblib`, which may execute code when loaded: only evaluate **trusted local artifacts**. This is a source-integrity check, not a guarantee against malicious artifacts.

It writes:

- classification report
- confusion matrix
- DENIED precision, recall, F1, and F2
- ROC-AUC
- average precision
- row-level DENIED probabilities and predictions (without patient/claim identifiers)
- always-PAID and always-DENIED baseline metrics, patient-grouped split provenance, and source fingerprint

## Model-selection logic

Each candidate model is trained only on the training split. Threshold selection uses validation data only. The best model is selected using validation average precision, with DENIED F2 as the tie-break. This is an engineering choice, **not an insurer-approved cost or staffing policy**. Performance should also be compared at fixed review capacity, with calibration, temporal holdout and patient/provider subgroup analysis.

The test split is used only after model and threshold selection are locked.

## Why threshold tuning matters

A default threshold of `0.50` is not automatically optimal for every business objective. In this project, a missed denied claim is treated as a more important error than an unnecessary review. F2 therefore weights recall more heavily than precision while still penalizing excessive false positives.

This framing is transferable to many risk-sensitive binary-classification problems beyond insurance.

## Tests

```bash
pytest -q
```

## Original statistical analysis

The notebook also contains hypothesis-testing work, including one-sample and two-sample t-tests, ANOVA, and chi-square analysis. Those analyses remain part of the exploratory/statistical component; the upgraded `src/` package focuses on the reproducible ML workflow.

## Limitations

- There are no independently reproduced results for the new patient-grouped pipeline; the historical notebook's older split and threshold results are not interchangeable with these tests.
- The model uses a compact feature set and does not establish causal relationships.
- The current F2 operating objective is illustrative; real adoption needs cost estimates and review capacity. A grouping-safe split does not replace a future-time holdout.
- Predictors such as claim/visit costs must be verified as available **before** a contemplated review decision, to avoid outcome leakage. Real deployments also require calibration, drift monitoring, fairness/subgroup checks, data contracts, privacy approval and examiner workflows.
