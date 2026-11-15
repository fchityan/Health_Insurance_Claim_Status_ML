# Model Validation, Claims Governance & Operational Assurance

**Current evidence:** documented model scripts, patient-grouped split, source-integrity checks and synthetic-data regression tests; no source CSVs or verified current numerical performance published. **No clinical/insurance production approval, independent privacy review or completed claims UAT is asserted.**

## Assurance plan and UAT evidence register
| Verification | Evidence required before pilot | Current state |
|---|---|---|
| Data approval | Permitted datasets, data dictionary, sensitive-field inventory and join audit | Not published |
| Label validity | Manual claims-status adjudication samples, reversal and delayed-label checks | Proposed |
| Model leakage | Patient-grouped split and locked source SHA-256; independent temporal and feature-timing audit | Group isolation implemented; time-based review outstanding |
| Reproducibility | Training command, seeds, source fingerprint, grouped split, test runner, independently reproduced metrics | Synthetic regression suite added; source-data metrics and dependency lock outstanding |
| Robustness | Missed-denial and false-positive costs, calibration and subgroup error | Not independently evidenced |
| Data privacy | Role controls, encryption, minimal use, retention and deletion tests | Proposed |
| Business UAT | Examiner override, missing claim, wrong match, appeal, operational queue | Not executed/recorded |
| Release/operations | Monitoring, audit export, kill switch, rollback and incident triage | No deployed workflow evidenced |

## Risk register and mitigation
| Risk | Consequence | Proposed mitigation |
|---|---|---|
| Wrong multi-table join | Misleading claim score and customer harm | Referential checks, row-level reconciliation and manual audit |
| Future payment status leakage | Inflated test results | As-of feature availability and temporal holdout |
| Differential impact | Unfair claims attention | Subgroup analysis, fairness review and human appeal |
| Sensitive health/claim data disclosure | Serious privacy harm | Least privilege, data minimisation, TLS/encryption, retention, private logging |
| Score mistaken for adverse decision | Unfair automatic claim handling | Mandatory examiner decision, override and reason trail |
| Drift / changed adjudication policy | Invalid model recommendations | Policy change gate, ongoing outcome monitoring and retraining approval |

## Security, privacy and human oversight
Create a data-protection impact assessment where required by the applicable jurisdiction, capture lawful basis/contractual authority, access reviews, secure development logs, retention and deletion standards, secret/credential rotation, threat model for inference and storage, and a tested incident-notification process. These controls are **design requirements** until evidenced.

**Proposed RACI:** claims sponsor approves use case and cost threshold; examiner lead owns human review; data owner controls source & labels; model owner validates; security/privacy officer reviews access; platform owner runs releases and incidents. No named/approved RACI exists.

**Incident scenario:** false escalation, misjoined claim, privacy event, model drift or service outage → suspend risk-based routing → revert to manual claims rules → preserve protected evidence → isolate release/source → risk-owner review → revalidate → document corrective action and release approval.

**Go/no-go:** dataset authorization, independent test report, examiner UAT signoff, fairness review, privacy/security assessment, monitoring SLA, incident and rollback exercises, accountable ownership. Missing gates must be marked *not satisfied*, never silently inferred.
