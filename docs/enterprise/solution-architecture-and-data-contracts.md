# Claims Data Architecture, Feature Contract & Decision Support

**Status:** Scriptable research prototype; no real claims-system/API integration or approved review queue. A restricted two-feature intake model is implemented; **source-data timing, independently verified performance and any operational approval remain OPEN**.

```text
Authorised encounters / claims / patients / providers / organizations CSVs
  → strict claim/encounter join validation and mature PAID/DENIED labels only
  → pre-adjudication minimal profile: visit_type + age_at_service only
  → type checks; numeric median & categorical mode imputation
  → one-hot encoding and consistent train-only fit
  → five-fold patient-grouped splitting (approximately 60%/20%/20%)
  → Logistic Regression / Random Forest / Gradient Boosting
  → validation model ranking and DENIED F2 threshold
  → serialised model bundle + SHA-256 source fingerprint → locked evaluation
  → hypothetical claims-review queue (not implemented)
```

## Proposed data contract & lineage
| Entity / field family | Reference use | Required data-governance validation |
|---|---|---|
| Claim | Historical PAID/DENIED target and join provenance, **not** model costs | Define final outcome, reversals, status updates and label observation time |
| Encounter | Visit type predictor and service context; total visit cost **excluded** | Verify join cardinality, visit-type recording timestamp and pre-adjudication availability |
| Patient | Age at service only; gender **excluded** from the predictor set | Verify lawful basis, birthdate and service-date availability at intake |
| Provider | Organization/service metadata | Vendor/clinician permissions; missing providers and category drift |
| Organization | Contextual reference | Referential integrity and permissible joins |
| Prediction record | Denied risk class, probability/threshold and model/version | Avoid explanations implying coverage/clinical certainty |

**Prediction point:** Proposed after a claim is received and the encounter context is known, before a claims examiner makes an adjudication or updates status. This is **not verifiable** from the supplied service date alone. The `pre_adjudication_minimal_v1` model profile uses only `visit_type` and `patient_age`; `claim_amount`, `total_visit_claim_cost` and gender are blocked as predictors. Any future additions require a versioned feature-contract change and external timestamp validation.

**Dataset boundary:** source CSVs are not committed. Actual canonical column/type definitions must be read from `src/pipeline.py`, approved by a data owner and tested on permitted data; this table is **not** a signed physical schema.

## Architectural decisions
- **ADR-HC-01:** Use denied-class F2 because missing likely denials is treated as an important *review-priority* error; cost weights remain hypothetical.
- **ADR-HC-02:** Select model/threshold only on training/validation and preserve untouched test.
- **ADR-HC-03:** Use disjoint patient groups across train/validation/test via stratified five-fold splitting; this reduces repeat-patient leakage but is **not** an out-of-time evaluation. Independently verify feature availability as of a real decision timestamp.
- **ADR-HC-04:** Model scores are not adjudication authority; operational human review is mandatory for real decisions.
- **ADR-HC-05:** Claims and health-related features require minimisation, privacy review, protected storage and controlled audit access before usage.
- **ADR-HC-06:** Fail closed on non-final outcome labels; reject unapproved legacy feature artifacts. No local training output qualifies as independently validated evidence.

## Integration design still to be decided
Claims intake/queue API, source freshness, claim amendments, prediction timing, reviewer explanation UX, integration retries, idempotent re-scoring, batch limits, audit events, role-based access, appeal/override and feedback capture. No deployed service is asserted.


For data-custodian and performance evidence requirements see [P0 evidence and data gates](p0-evidence-gates.md).
