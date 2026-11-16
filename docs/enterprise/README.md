# Insurance Claims Analytics — Solution Design and Governance

This collection brings together the **business, technical and governance artefacts** for a historical denial-risk screening research prototype. The proposed outcome is an aid to claims-examiner prioritisation, not an automated decision about coverage, payment or fraud.

## Documentation library

| Workstream | Document | Focus |
|---|---|---|
| Business analysis and requirements | [Business requirements and traceability](business-requirements-and-traceability.md) | Proposed operating problem, stakeholders, decision rights and acceptance criteria |
| Solution architecture and data design | [Architecture and data contracts](solution-architecture-and-data-contracts.md) | Data lineage, conservative feature scope, technical design and integration considerations |
| Model validation and operational assurance | [Assurance, security and operations](assurance-security-and-operations.md) | Performance-validation approach, privacy, human review, controls and operational readiness |
| Validation and risk management | [Validation and risk register](validation-gap-register.md) | Implemented safeguards, outstanding evidence and acceptance prerequisites |

## Implementation references

The corresponding software is available in the [data pipeline](../../src/pipeline.py), [training workflow](../../src/train.py), [evaluation workflow](../../src/evaluate.py) and [synthetic regression tests](../../tests/test_integrity.py).

**Project maturity:** The pipeline and automated tests are implemented. Independent results on authorised historical data, a verified prediction-time feature audit, operational examiner testing and production approvals have not been demonstrated. All business and integration materials describe a proposed approach, not an insurer-approved production solution.
