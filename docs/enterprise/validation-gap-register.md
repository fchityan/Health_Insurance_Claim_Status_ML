# Validation Gap Register — Health Insurance Claim Status ML

**Scope:** Historical PAID/DENIED classification for a possible *human review-prioritisation* use case. No policy-compliant claims adjudication or fraud outcome is inferred from these labels.

**Evidence key:** Implemented = code/config present on this branch; Tested = synthetic regression test written, CI execution separately verifiable; Open = requires source data, independent validation, organizational decision or integration.

| Priority | Risk or gap | Current status | Evidence / next validation |
|---|---|---|---|
| P0 | Claim/encounter joins silently duplicate or mismatch patient/provider/organization records | **Implemented + synthetic tests** | `src/pipeline.py`; orphan, duplicate, mismatch, and multi-claim tests in `tests/test_integrity.py` |
| P0 | Repeated patients leak between training and holdout | **Implemented + synthetic tests** | Five-fold stratified patient-group split. Does **not** substitute for out-of-time validation |
| P0 | Evaluator uses a different or reordered source dataset from training | **Implemented + synthetic tests** | Source CSV SHA-256 stored with the trusted model; changed input blocks evaluation |
| P0 | Claims model interpreted as deciding which claims should be denied | **Clarified; business review still open** | README defines historic outcome, not coverage entitlement, medical necessity or fraud |
| P0 | Claim and encounter costs might not exist at prediction cutoff | **Open** | Establish decision-time data availability and run a leakage audit/feature ablation |
| P0 | Source CSVs absent; current test metrics not independently reproduced | **Open** | Obtain authorized data, preserve provenance and rerun grouped and temporal holdouts |
| P1 | High DENIED recall can result from flagging everything DENIED | **Baseline reporting implemented** | Always-PAID and always-DENIED metrics added; compare review workload and precision |
| P1 | Multiple versions of historical claim status, appeals and reversals | **Open** | Define label-finalization policy and outcome time window with claims owners |
| P1 | Stratified patient groups do not prevent future-to-past leakage | **Open** | Construct out-of-time test with adequate outcome maturity and no shared patient leakage |
| P1 | Threshold is tuned to F2 rather than insurer costs or examiner capacity | **Open** | Compare cost-weighted expected value, precision at review capacity, calibration and workload |
| P1 | Gender/age/provider-related proxy effects may create inequitable review | **Open** | Sensitive-attribute review, subgroup error, drift, appeal and override procedures |
| P1 | Workflow does not integrate with review queues or capture examiners' dispositions | **Open** | Prototype explanation, reviewer overrides, audit events and UAT with deidentified cases |
| P1 | Handling of health/claim information needs independent privacy and security review | **Open** | Data minimization, approved collection, encryption, access/retention policy and assessment |
| P2 | Version pins, reproducible environments, deployment monitoring and incident response | **Partial** | CI regression workflow added; lock dependencies and produce runtime/operations evidence |

## Release gates (not signed off)

1. Business sponsor signs a **review-prioritisation** use case, claim-status glossary, cutoff time, eligibility and measurable human-review objective.
2. Data steward verifies source rights, join lineage, status maturity, patient-group independence, missingness and decision-time feature availability.
3. Independent reviewer reproduces grouped and **out-of-time** held-out precision/recall, PR-AUC, baseline comparison, calibration and subgroup error on authorized data.
4. Claims operations signs realistic examiner UAT, capacity/false-alarm policy, override procedures and appeals.
5. Security/privacy and model-risk reviewers accept access, retention, audit and incident controls.

**Outcome:** This branch improves code integrity and testing, but none of these release gates are implied to have passed. The older notebook's reported results must not be promoted as current grouped-test metrics.
