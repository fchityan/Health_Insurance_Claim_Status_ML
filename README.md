# Health Insurance Claim Status ML

This project analyzes health insurance claim behavior in two parts:

1. Statistical fraud-risk assessment using hypothesis testing.
2. Binary machine learning classification to predict whether a claim is `PAID` or `DENIED`.

The full analysis is implemented in the notebook:
- `Health_Insurance_Claim_Status_ML.ipynb`

## Project Objectives

### Part 1: Statistical Risk Assessment
- Compare a selected provider's claim-cost behavior against peer providers in the same specialty.
- Compare claim costs between two organizations.
- Compare claim costs across multiple organizations in selected cities.
- Compare encounter-class distributions between providers.

Main statistical tests used:
- One-sample t-test
- Independent two-sample t-test
- One-way ANOVA
- Chi-square test of independence

### Part 2: Machine Learning for Claim Status
- Build a binary classifier for `PAID` vs `DENIED` claims.
- Prioritize business risk reduction by minimizing missed denied claims.
- Evaluate baseline models and perform threshold tuning.

Models used:
- Random Forest Classifier
- Gradient Boosting Classifier

## Data Files Expected

The notebook loads the following CSV files from the project root:
- `encounters.csv`
- `claims.csv`
- `organizations.csv`
- `patients.csv`
- `providers.csv`

If these files are not present, the notebook will fail at data-loading steps.

## Feature Engineering and Modeling Workflow

The notebook performs the following high-level steps:

1. Load and clean core datasets.
2. Rename columns for consistency across tables.
3. Join encounters, claims, and patient/provider/organization context.
4. Convert patient birthdate into numeric age.
5. Filter claim statuses to binary classes (`PAID`, `DENIED`).
6. Build preprocessing with:
   - `OneHotEncoder` for categorical features
   - Numeric passthrough for numeric columns
   - `ColumnTransformer` + `Pipeline`
7. Split dataset into train/validation/test using a 60/20/20 strategy.
8. Train and compare Random Forest and Gradient Boosting.
9. Evaluate with classification reports and confusion matrices.
10. Tune decision threshold (0.40 used in notebook) to improve recall for denied claims.
11. Inspect feature importance for model interpretability.

## Key Findings in Notebook

### Statistical Analysis Highlights
- Provider-vs-peer one-sample t-test: p-value reported as `0.099`, so no significant evidence at alpha `0.05` that the selected provider's mean claim cost is higher than peers.
- Organization-vs-organization two-sample t-test: p-value reported as approximately `0.692`, indicating no significant difference in mean claim costs.
- Provider encounter-mix chi-square test: p-value reported as `0.263`, suggesting no significant dependency between provider identity and encounter-class distribution at alpha `0.05`.

### Machine Learning Highlights
- The project frames missed denied claims as the primary business risk.
- Threshold tuning below `0.5` is used to improve denied-claim recall.
- A threshold of `0.40` is selected in the notebook to increase denied recall, with expected precision/accuracy trade-offs.

## Environment Setup

Use Python 3.9+ recommended.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib scipy scikit-learn jupyter
```

## How to Run

1. Place required CSV files in the project root.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Open `Health_Insurance_Claim_Status_ML.ipynb`.
4. Run cells from top to bottom.

## Repository Structure

```text
.
├── Health_Insurance_Claim_Status_ML.ipynb
└── README.md
```

## Notes and Limitations

- This repository currently centers on a single notebook workflow.
- Data files are not included in this repository structure view.
- Reproducibility depends on having matching data schemas for the expected CSV files.

## Suggested Next Improvements

- Add a `requirements.txt` with pinned versions.
- Add a data dictionary describing input columns.
- Add a scriptable pipeline (`src/`) for train/evaluate outside notebooks.
- Add ROC/PR analysis and threshold optimization by explicit business cost matrix.
- Add model persistence and inference script for production-like scoring.
