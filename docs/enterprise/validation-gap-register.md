# Model Validation and Risk Register — Historical Denial-Risk Screening

**Scope:** Historical PAID/DENIED classification for a possible *human review-prioritisation* use case. No policy-compliant claims adjudication or fraud outcome is inferred from these labels.

**Evidence key:** Closed (code-control scope) = the defined technical safeguard is implemented and regression-tested; Controlled / Open = protective code or documentation exists but independent validation or approval is still outstanding; Open = the underlying business/data evidence is not yet available. A technical control closure is **not** model validation, risk acceptance or permission to deploy.

**Critical closure summary:** **3 of 6 technical risks closed within the tested code scope; 3 of 6 critical risks remain open or only partly controlled.** This summary concerns the current research prototype, not operational readiness.

| Priority | Risk or gap | Current status | Evidence / next validation |
|---|---|---|---|
| Critical | Claim/encounter joins silently duplicate or mismatch patient/provider/organization records | **Closed — code-control scope only** | `src/pipeline.py` validates relationships; orphan, duplicate, mismatch and multi-claim synthetic regression tests in `tests/test_integrity.py`. Real source-data reconciliation remains an external validation gate |
| Critical | Repeated patients leak between training and holdout | **Closed — code-control scope only** | Disjoint patient groups are enforced and synthetically tested. Out-of-time, provider and policy-period leakage are **not** resolved by grouping |
| Critical | Evaluator uses a different or reordered source dataset from training | **Closed — code-control scope only** | Source-file SHA-256 stored and verified; modified CSV input is rejected in synthetic regression tests. This is not a source-provenance or label-quality certification |
| Critical | Historical status confused with policy-correct adjudication | **Controlled / OPEN — business definition and approval** | Documentation restricts purpose to proposed human review prioritisation. Outcome provenance, reversals, final-status glossary, and claims-owner sign-off have not been evidenced |
| Critical | Unavailable, post-decision or inappropriate predictors affect modelling | **Controlled / OPEN — point-in-time evidence** | Default features are limited to visit type and age, with cost and gender excluded and tested. A data steward must verify both retained features existed at the proposed intake cutoff |
| Critical | Source CSVs absent; current test metrics not independently reproduced | **OPEN — external dataset and independent results** | Obtain authorised, appropriately matured claims data; independently reproduce grouped and out-of-time holdouts and report precision, recall, PR-AUC, baselines and review capacity. Synthetic CI is not evidence of predictive usefulness |
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


## What would close the remaining critical risks

- **Historical outcome definition:** Claims owner approves which status counts as final, the timing of reversals/appeals, and the fact that the prediction is only for review priority. Documentation alone does not close this risk.
- **Point-in-time validity:** A data steward supplies event timestamps or equivalent auditable evidence that visit type, service date and permitted birthdate were available by the proposed scoring time, with no label or downstream information leakage.
- **Verified model performance:** An authorised historical cohort is evaluated with mature labels, a locked model and holdout; an independent reviewer reproduces metrics and a later-period test, comparing against fixed-capacity operational baselines. If the two-feature model adds no signal, the appropriate outcome is to discontinue the classifier or redefine the project, not mark performance as passed.

**No production release gate has been signed off.** The current software tests support technical-control closure only.
