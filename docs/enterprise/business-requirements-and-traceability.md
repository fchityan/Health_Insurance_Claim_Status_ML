# Business Requirements & Process Traceability — Historical Denial-Risk Screening

**Status:** Enterprise-style **draft** for portfolio review; no insurer/customer signoff, completed UAT, regulatory assurance or production owner is asserted.

## Problem, users and human decision rights
The supervised label is a **historically recorded** `PAID` or `DENIED` status. The proposed business purpose is to prioritise some **received claims for earlier human examination** before adjudication. The model cannot decide whether a claim ought to be paid, whether policy coverage exists, or whether any provider or claimant committed fraud.

**Proposed decision cutoff:** After claim receipt and encounter metadata availability, **before** initial adjudication, settlement, appeals or status changes. No claim-intake or attribute-update timestamps are available in this codebase to prove the cutoff; source-owner validation remains a P0 blocker.

**Decision rights:** The examiner follows policy, may disregard/override the score, and records the eventual review outcome. No automatic adverse action is permitted.

**Potential stakeholders:** claims operations manager, claim examiner, policy administration specialist, data steward, information security/privacy officer, model-risk owner, and business analyst.

**Illustrative As-Is:** claims are investigated through manual rules and operational queues. **Proposed To-Be:** validate claim data → compute human-review risk indicator → route flagged cases to an examiner → capture reasoned decision/outcome → monitor effectiveness and fairness. This operational queue is *not* implemented or verified in the repository.

## Business objectives and KPI hypotheses
At a **fixed number of examiner reviews**, compare historical-denied identification against ordinary queue ordering and naive class baselines; also measure false escalations, review time, preventable rework, claim timeliness, customer complaints and permitted-subgroup effects. **No improvements have been measured or claimed.** Need a baseline, target and explicit case mix before any go/no-go decision.

## Proposed requirements matrix
| ID | Priority | Requirement | Repo anchor / evidence | Proposed acceptance |
|---|---|---|---|---|
| BR-01 | Must | Model's role is human review prioritisation | README and model workflow | Examiner can reject or override suggestion |
| BR-02 | Must | Define final historical `PAID` / `DENIED` labels and outcome maturity | Non-final/unknown values now fail closed; actual status lineage unverified | Business owner approves status glossary, reversals and outcome window |
| FR-01 | Must | Assemble encounters, claims, patients, providers and organizations | `src/pipeline.py` validates ID uniqueness, join cardinality and matching keys | Test duplicate IDs, orphan claims, conflicting patient/provider/organization IDs and multi-claim encounters |
| FR-02 | Must | Exclude unapproved cost, demographic and post-adjudication predictors | **Implemented** two-feature intake profile; source event timestamps still unverified | Test excluded fields and obtain data-steward cutoff attestations |
| FR-03 | Must | Impute missing values, encode categories, isolate patient groups | Preprocessing and five-fold grouped split | No patient overlap across partitions; both claim outcomes in each partition |
| FR-04 | Must | Compare LR, Random Forest, Gradient Boosting | `src/train.py` | Reproduced model comparison with source-data approval |
| FR-05 | Must | Tune denied-class threshold on validation F2 | Training logic and source fingerprint | Test set never tunes threshold and altered source files invalidate evaluation |
| NFR-01 | Must | Limit access to personal/claim-level data and predictions | **Production controls not evidenced** | Privacy impact assessment and negative access tests |
| NFR-02 | Must | Provide review history, model version and fallbacks | **Queue/workflow not implemented** | Runbook and UAT execution |
| NFR-03 | Should | Check calibration, subgroup performance and drift | Listed as future limitation | Model-risk acceptance report |

## Decisions and approvals to obtain
Claims definition and review cost; lawful data usage; minimum necessary input fields; model decision rights; denial-warning communication; independent validation metrics; customer impact; reviewer override; retention/deletion; model change authority. These approvals are **not recorded**.

**Business acceptance is distinct from model accuracy:** reviewers should test false alarms, incorrect joins, aged claims, incomplete records, explainability and escalation behaviour before any live pilot.


See [P0 evidence and data gates](p0-evidence-gates.md) for the external checks required before any insurer pilot or performance claim.
