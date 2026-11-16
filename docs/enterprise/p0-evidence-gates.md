# P0 Evidence Gates — Historical Denial-Risk Screening for Human Review

**State as of 16 November 2026: code controls improved; empirical and organisational gates NOT PASSED.**

This is a research-model checklist, not permission to process real claims or deny benefits. The target is a historically recorded PAID/DENIED status; it is not a claim entitlement, medical necessity, fraud, coverage or claims-decision label.

## 1. Proposed operational decision

**When:** After an authorised claim intake event has captured an encounter classification and the dates needed to calculate age at service, **before** claims examination or adjudication. **Important:** the current CSV source records a service date but does not demonstrate a claim-receipt timestamp, a feature-created-at timestamp, or when historical payment status became final. The "as-of" decision point cannot be verified from the repository.

**Who:** A human claims examiner; any future score is only a possible indication to review a claim earlier in the queue. All decisions remain with the examiner under existing policy.

**What is success:** At an **equal, pre-agreed number of cases reviewed**, identify more historically denied cases than a baseline queue while tracking unnecessary extra reviews, time and fairness. An insurer must agree whether this is actually valuable and what an acceptable false-positive burden means.

## 2. Known leakage containment — code complete, temporal audit pending

| Column or field | Default use | Reason |
|---|---|---|
| `visit_type` | **Model feature** | May be known at intake; timestamps still unverified |
| `patient_age` | **Model feature** | Derived from service date and birthdate; permission and availability require approval |
| `claim_amount` | **Not a feature** | Could be incomplete or amended after claim receipt |
| `total_visit_claim_cost` | **Not a feature** | Could arise from post-service billing or downstream claim processing |
| `patient_gender` | **Not a feature** | Excluded by default for minimisation and proxy-risk reduction |
| `PAID` / `DENIED` claim status | **Training target only** | Must be a documented, mature historical outcome, never a feature |
| Patient/claim IDs and service date | **Grouping/provenance only** | Not model predictors |

Unapproved and unknown status values now fail closed. Previous model artifacts are invalidated by an explicit feature-profile contract. The choice of two conservative features makes the problem harder; it does **not** establish that the risk is predictable from those features.

**Still mandatory:** a field-by-field extraction-time audit, labelled event lineage, missingness audit and verification that visit type and dates were available by the proposed prediction cutoff. Grouped folds address repeat-patient leakage but do not rule out temporal changes, provider leakage or evolving policy practices.

## 3. Empirical evidence that is NOT available yet

The current repository contains no authorised original source CSVs, no independently reproduced updated source-data holdout numbers and no test on later claim periods. A synthetic CI success verifies *software behaviour*, not real-world usefulness.

For an authorised dataset, the controlled evidence sequence is:

1. Data steward records provenance, source rights, extraction cutoff, schema, access controls, deidentification policy, label meaning, update/reversal rules and outcome maturity. Record counts by label and dates, without uploading records.
2. Confirm that the two model features were available **before** any historical decision. If not, stop; do not train.
3. Retrain using `python -m src.train --data-dir <approved-directory>`. Produce model-comparison evidence and record source hashes and package versions.
4. Evaluate locked grouped holdout with `python -m src.evaluate --data-dir <approved-directory>`; report sample sizes, class prevalence, confusion matrix, precision, recall, F2, PR-AUC, ROC-AUC and both trivial baselines.
5. Independently repeat on a controlled copy of authorised input and evaluate **later-period claims** (with labels sufficiently final). An out-of-time split must not select a threshold on the test interval.
6. Calculate operational metrics at a **fixed review capacity**: cases reviewed, denied cases identified and incorrect escalations. Include a sensitivity analysis across capacity and outcome prevalence. Explain any trade-off in time, service, fairness and human review.
7. Seek signed-off business, privacy, model-risk and examiner testing decisions. Until then, state research-only, regardless of achieved metrics.

**Do not report historical notebook scores as results of the safer pipeline.** Do not report any model score as an actual reduction in claim denials, lower costs or fraud detection without the corresponding verified outcome study.

## 4. Evidence release table

| Gate | Current status | Required proof |
|---|---|---|
| Referential integrity, patient separation, source hashes | Implemented + synthetic tests | CI run and reviewer inspection |
| Conservative default feature profile | Implemented + synthetic tests | Source diff and regression tests |
| Label-filter safety | Implemented + synthetic tests | Unknown/pending status fails closed |
| Feature availability as of claim intake | **OPEN** | Historical source timestamps/data-owner attestation |
| Label provenance, reversal/appeal maturity | **OPEN** | Approved final-outcome glossary, timestamps and sample audit |
| Reproducible updated performance on authorised data | **OPEN** | Signed sample lineage, independently repeated metrics |
| Later-period/generalisation test | **OPEN** | Mature outcomes from future time interval |
| Operational value at fixed review capacity | **OPEN** | Explicit baselines, costs, queue capacity and reviewer acceptance |
| Legal basis, privacy, security, review governance | **OPEN** | Independent approval and documented operational controls |

## 5. Interpret the test outputs correctly

The evaluation JSON deliberately includes:
- `evidence_status: unverified_local_holdout_research_only`
- `independently_reproduced: false`
- `out_of_time_validated: false`
- `deployment_approved: false`

These fields must not be changed merely because a local metrics file exists. Independent and insurer approvals require separate evidence.

**Bottom line:** P0 *code-level safeguards* can be implemented now. P0 *real data validity, business sign-off, and independently verified performance* cannot be completed from the repository alone.
