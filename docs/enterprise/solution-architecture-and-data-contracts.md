# Claims Data Architecture, Feature Contract & Decision Support

**Status:** Scriptable ML prototype; no claim-system/API integration verified.

```text
Authorised encounters / claims / patients / providers / organizations CSVs
  → multi-table assembly and deterministic age-at-service feature
  → type checks; numeric median & categorical mode imputation
  → one-hot encoding and consistent train-only fit
  → stratified 60% train / 20% validation / 20% locked test
  → Logistic Regression / Random Forest / Gradient Boosting
  → validation model ranking and DENIED F2 threshold
  → serialised model bundle → post-selection test report
  → hypothetical claims-review queue (not implemented)
```

## Proposed data contract & lineage
| Entity / field family | Reference use | Required data-governance validation |
|---|---|---|
| Claim | Binary payment/denial target and claim context | Define final outcome, reversals, status updates and label observation time |
| Encounter | Service context | Confirm many-to-one joins, service timestamp and pre-decision availability |
| Patient | Demographics; age derived from service date | Minimise identifier exposure, verify allowed subgroup use and consent/legal basis |
| Provider | Organization/service metadata | Vendor/clinician permissions; missing providers and category drift |
| Organization | Contextual reference | Referential integrity and permissible joins |
| Prediction record | Denied risk class, probability/threshold and model/version | Avoid explanations implying coverage/clinical certainty |

**Dataset boundary:** source CSVs are not committed. Actual canonical column/type definitions must be read from `src/pipeline.py`, approved by a data owner and tested on permitted data; this table is **not** a signed physical schema.

## Architectural decisions
- **ADR-HC-01:** Use denied-class F2 because missing likely denials is treated as an important *review-priority* error; cost weights remain hypothetical.
- **ADR-HC-02:** Select model/threshold only on training/validation and preserve untouched test.
- **ADR-HC-03:** Group related claim events by appropriate patient/claim/time where needed; independently evaluate whether stratified record-level splits cause entity or temporal leakage.
- **ADR-HC-04:** Model scores are not adjudication authority; operational human review is mandatory for real decisions.
- **ADR-HC-05:** Claims and health-related features require minimisation, privacy review, protected storage and controlled audit access before usage.

## Integration design still to be decided
Claims intake/queue API, source freshness, claim amendments, prediction timing, reviewer explanation UX, integration retries, idempotent re-scoring, batch limits, audit events, role-based access, appeal/override and feedback capture. No deployed service is asserted.
