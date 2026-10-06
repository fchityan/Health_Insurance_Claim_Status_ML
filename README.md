# Cost-Sensitive Claim Status Classification

A reproducible binary-classification project for predicting whether a claim is `PAID` or `DENIED`, with emphasis on decision-threshold optimization, model comparison, and reducing costly missed denials.

The original notebook, `Health_Insurance_Claim_Status_ML.ipynb`, is preserved. The repository now includes a stronger scriptable ML workflow designed to showcase reproducible modeling and production-oriented engineering rather than notebook-only analysis.

## What the project demonstrates

- Multi-table feature assembly from encounters, claims, patients, providers, and organizations
- Deterministic patient-age engineering using service date rather than the current calendar year
- Missing-value handling through model pipelines instead of dropping incomplete rows
- Numeric median imputation and categorical most-frequent imputation
- One-hot encoding for categorical variables
- Stratified 60/20/20 train/validation/test splitting
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Validation-only threshold tuning for the `DENIED` class
- Recall-weighted F2 scoring for cost-sensitive classification
- Probability-level evaluation using ROC-AUC and average precision
- Saved model artifacts and reproducible holdout evaluation
- Unit tests for threshold selection, preprocessing, and deterministic feature engineering

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

The data files are not committed to the repository.

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

This evaluates the selected model and threshold on the untouched test split and writes:

- classification report
- confusion matrix
- DENIED precision, recall, F1, and F2
- ROC-AUC
- average precision
- row-level DENIED probabilities and predictions

## Model-selection logic

Each candidate model is trained only on the training split. Threshold selection is performed only on the validation split. The best model is selected using validation average precision, with DENIED F2 as the tie-break.

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

- Exact metrics depend on the source dataset and should be regenerated before publication.
- The model uses a compact feature set and does not establish causal relationships.
- Cost-sensitive threshold selection should ideally be tied to quantified operational costs.
- Real deployments would require calibration, drift monitoring, subgroup checks, data contracts, and human-review workflows.
