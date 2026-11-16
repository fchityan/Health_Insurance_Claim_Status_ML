# Model Validation and Risk Register — Historical Denial-Risk Screening

**Scope:** Historical PAID/DENIED classification for a possible *human review-prioritisation* use case. No policy-compliant claims adjudication or fraud outcome is inferred from these labels.

**Evidence key:** Implemented = code/config present on this branch; Tested = synthetic regression test written, CI execution separately verifiable; Open = requires source data, independent validation, organizational decision or integration.

| Priority | Risk or gap | Current status | Evidence / next validation |
|---|---|---|---|
| Critical | Claim/encounter joins silently duplicate or mismatch patient/provider/organization records | **Implemented + synthetic tests** | `src/pipeline.py`; orphan, duplicate, mismatch, and multi-claim tests in `tests/test_integrity.py` |
| Critical | Repeated patients leak between training and holdout | **Implemented + synthetic tests** | Five-fold stratified patient-group split. Does **not** substitute for out-of-time validation |
| Critical | Evaluator uses a different or reordered source dataset from training | **Implemented + synthetic tests** | Source CSV SHA-256 stored with the trusted model; changed input blocks evaluation |
| Critical | Historical status confused with policy-correct adjudication | **Use case narrowed; business approval OPEN** | Intake-stage prioritisation for human examiner only; historical status is not entitlement, fraud or payment recommendation |
| Critical | Claim and visit costs, gender or post-decision variables leak into model features | **Default excluded + synthetic tests; as-of audit OPEN** | `pre_adjudication_minimal_v1` restricts model to visit type and age; require evidence that both were available by claim intake |
| Critical | Source CSVs absent; current test metrics not independently reproduced | **OPEN** | Obtain authorised, mature-status data; independently reproduce grouped and later-period holdouts. Synthetic CI is not a model-performance result |
| High | High DENIED recall can result from flagging everything DENIED | **Baseline reporting implemented** | Always-PAID and always-DENIED metrics added; compare review workload and precision |
| High | Multiple versions of historical claim status, appeals and reversals | **OPEN; non-final values fail closed** | Explicit mature PAID/DENIED-only cohort enforced; status-finalisation policy and outcome window need claims-owner approval |
| High | Stratified patient groups do not prevent future-to-past leakage | **Open** | Construct out-of-time test with adequate outcome maturity and no shared patient leakage |
| High | Threshold is tuned to F2 rather than insurer costs or examiner capacity | **Open** | Compare cost-weighted expected value, precision at review capacity, calibration and workload |
| High | Gender/age/provider-related proxy effects may create inequitable review | **Open** | Sensitive-attribute review, subgroup error, drift, appeal and override procedures |
| High | Workflow does not integrate with review queues or capture examiners' dispositions | **Open** | Prototype explanation, reviewer overrides, audit events and UAT with deidentified cases |
| High | Handling of health/claim information needs independent privacy and security review | **Open** | Data minimization, approved collection, encryption, access/retention policy and assessment |
| Medium | Version pins, reproducible environments, deployment monitoring and incident response | **Partial** | CI regression workflow added; lock dependencies and produce runtime/operations evidence |

## Release gates (not signed off)

1. Business sponsor signs a **review-prioritisation** use case, claim-status glossary, cutoff time, eligibility and measurable human-review objective.
2. Data steward verifies source rights, join lineage, status maturity, patient-group independence, missingness and decision-time feature availability.
3. Independent reviewer reproduces grouped and **out-of-time** held-out precision/recall, PR-AUC, baseline comparison, calibration and subgroup error on authorized data.
4. Claims operations signs realistic examiner UAT, capacity/false-alarm policy, override procedures and appeals.
5. Security/privacy and model-risk reviewers accept access, retention, audit and incident controls.

**Implemented model safeguards:** The maintained pipeline rejects unknown/non-final labels and legacy feature artifacts, and excludes claim/visit-cost predictors and gender by default. These controls reduce known risks but do **not** verify historical feature timing or current model performance. See [Model assurance and operations](assurance-security-and-operations.md).

**Outcome:** Code integrity has improved, but none of the external release gates are implied to have passed. The historical notebook's results cannot be promoted as current grouped-test metrics.
