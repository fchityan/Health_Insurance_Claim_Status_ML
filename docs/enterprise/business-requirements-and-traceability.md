# Business Requirements & Process Traceability — Claim Denial Risk

**Status:** Enterprise-style **draft** for portfolio review; no insurer/customer signoff, completed UAT, regulatory assurance or production owner is asserted.

## Problem, users and human decision rights
A binary model predicts whether a claim will be `PAID` or `DENIED`. The useful business purpose is **early review prioritisation** of cases that may be denied—not automated denial, payment, coverage decisions or medical recommendations.

**Potential stakeholders:** claims operations manager, claim examiner, policy administration specialist, data steward, information security/privacy officer, model-risk owner, and business analyst.

**Illustrative As-Is:** claims are investigated through manual rules and operational queues. **Proposed To-Be:** validate claim data → compute human-review risk indicator → route flagged cases to an examiner → capture reasoned decision/outcome → monitor effectiveness and fairness. This operational queue is *not* implemented or verified in the repository.

## Business objectives and KPI hypotheses
Measure denied-claim recall at fixed examiner capacity; false escalation rate; review time; preventable rework; claim resolution timeliness; customer complaints; differential impact by permitted subgroups. **No improvements have been measured or claimed.** Need a baseline, target and explicit case mix before any go/no-go decision.

## Proposed requirements matrix
| ID | Priority | Requirement | Repo anchor / evidence | Proposed acceptance |
|---|---|---|---|---|
| BR-01 | Must | Model's role is human review prioritisation | README and model workflow | Examiner can reject or override suggestion |
| BR-02 | Must | Define policy-consistent `PAID` / `DENIED` label and timing | Multi-table source schema | Business owner approves label glossary |
| FR-01 | Must | Assemble encounters, claims, patients, providers and organizations | `src/pipeline.py` | Verified joins and duplicate/one-to-many handling |
| FR-02 | Must | Prevent future-outcome and post-decision information leakage | Feature selection / service-date age engineering | Time-bound feature audit |
| FR-03 | Must | Impute missing values, encode categories, preserve split integrity | Preprocessing pipeline | Missing/novel input tests |
| FR-04 | Must | Compare LR, Random Forest, Gradient Boosting | `src/train.py` | Reproduced model comparison with source-data approval |
| FR-05 | Must | Tune denied-class threshold on validation F2 | Training logic | Test set never used for threshold |
| NFR-01 | Must | Limit access to personal/claim-level data and predictions | **Production controls not evidenced** | Privacy impact assessment and negative access tests |
| NFR-02 | Must | Provide review history, model version and fallbacks | **Queue/workflow not implemented** | Runbook and UAT execution |
| NFR-03 | Should | Check calibration, subgroup performance and drift | Listed as future limitation | Model-risk acceptance report |

## Decisions and approvals to obtain
Claims definition and review cost; lawful data usage; minimum necessary input fields; model decision rights; denial-warning communication; independent validation metrics; customer impact; reviewer override; retention/deletion; model change authority. These approvals are **not recorded**.

**Business acceptance is distinct from model accuracy:** reviewers should test false alarms, incorrect joins, aged claims, incomplete records, explainability and escalation behaviour before any live pilot.
