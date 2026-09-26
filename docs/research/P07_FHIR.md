# P07 FHIR Interoperable Read-Only Decision Layer

Status: **implementation grain SG-000010**

P07 makes FHIR a governed decision interface rather than a formatting demo.

```text
FHIR-native input != FHIR-format memorization
resource retrieval != clinical correctness
tool routing != permission to execute a clinical action
representation gain != serialization artifact
R4 compatibility != silent semantic conversion to R5
```

## Standards freeze

GAX uses **HL7 FHIR R5 5.0.0** as the P07 canonical published-specification reference.

HL7's September 2026 `6.0.0-snapshot1` is the first full R6 ballot and is explicitly published as a local/snapshot build rather than the permanent current published specification. P07 therefore does not target R6 snapshot semantics.

P07 can ingest explicitly declared R4 resources for compatibility experiments. R4 identity remains R4 in model-visible metadata and evidence manifests. The renderer does not claim semantic R4-to-R5 conversion.

## Renderer, not validator

`gaxbench.fhir` provides deterministic rendering for research. It is **not** a replacement for the HL7 validator, terminology service, profile validation, or implementation-guide conformance testing.

The renderer:

- requires a non-empty `resourceType`;
- rejects non-finite/non-JSON content;
- recursively checks nested `Bundle.entry.resource` objects when present;
- sorts object keys for the canonical representation;
- preserves array order;
- excludes top-level FHIR narrative `text` by default;
- exposes explicit representation controls.

## Representation controls

P07 defines four representations:

1. `canonical-structured` — deterministic key order, narrative excluded;
2. `canonical-with-narrative` — same canonicalization with top-level narrative retained;
3. `source-order-json` — strict JSON preserving input object insertion order;
4. `flat-text` — deterministic path/value rendering of the narrative-free canonical object.

A paper-level FHIR gain is invalid if it is merely a serialization/order effect. P08 must compare representations under matched examples and revisions.

## Read-only action boundary

`FHIRReadOnlyAction` intentionally has no write action kind.

Allowed v0.1 action families:

- `read`;
- `search`;
- `select-resource-type`;
- `sufficient`;
- `need-more-information`;
- `abstain`;
- `verify-support`.

Explicit `POST`, `PUT`, `PATCH`, or `DELETE` query templates fail validation.

P07 does not authorize medication orders, treatment changes, record mutation, autonomous diagnosis, triage, or emergency execution.

## GAXBench integration

`FHIRDecisionCase` converts into the existing `BenchmarkItem` contract. The model-visible state records:

- `fhir_source_version`;
- canonical reference version;
- representation name;
- representation revision;
- rendered resources.

Allowed read-only actions become ordinary GAXBench actions so existing probability, calibration, abstention, and selective-risk metrics remain reusable.

## MedAgentBench source freeze

Initial P07 source identity:

```text
repository: stanfordmlgroup/MedAgentBench
revision: 99260117137b09f04837a8c18d18a1107efa55ae
code license: MIT
```

The upstream setup also relies on an external FHIR server image and an externally downloaded `refsol.py`. GAX does not treat the repository's MIT code license as automatically licensing those separate artifacts.

P07 qualifies a **local/user-supplied normalized export protocol**. It does not require paid OpenAI, Vertex, GCP, or other founder-paid infrastructure.

## FHIR-AgentBench source freeze

Initial P07 source identity:

```text
repository: glee4810/FHIR-AgentBench
revision: bbb42909a5a7eb907d1cd91f72a560729e7037ea
repository license: CC BY 4.0
source FHIR release in upstream setup: R4
```

The upstream README uses MIMIC-IV FHIR demo data and a Google Cloud Healthcare R4 FHIR store. Those source-data/cloud terms remain separate from the repository license.

P07 requires FHIR-AgentBench exports to retain `R4` identity. It does not silently translate them to R5 and does not hard-depend on GCP.

## Normalized external export contract

`ExternalFHIRTask` binds:

- benchmark identity;
- exact frozen upstream revision;
- task/source IDs;
- split and task family;
- explicit source FHIR version;
- redistribution-safe resources supplied by the caller;
- caller-provided read-only allowed actions;
- optional gold routing action and sufficiency target;
- dataset revision/license/redistribution/access metadata.

P07 rejects `split=test` by default. Final test execution belongs to P08 after models, representations, thresholds, benchmark revisions, and statistical protocol are frozen.

## Data and PHI boundary

Public P07 qualification uses abstract CC0 synthetic content only.

It must contain no:

- real patient record;
- PHI;
- clinically meaningful treatment threshold;
- diagnosis rule;
- medication recommendation;
- patient-care guidance.

MIMIC-derived or other credentialed/restricted records are not copied into GAX merely because an upstream benchmark uses them.

## Compute rule

Founder-paid cloud compute is out of scope.

The P07 adapter contract is intentionally file/local-export based. Reference cloud environments may be documented as upstream provenance without becoming a requirement for GAX qualification.

## Paper gate

P07 implementation can establish that GAX has a reproducible, version-aware, read-only FHIR decision interface.

It cannot establish:

- clinical correctness;
- patient safety;
- EHR deployment readiness;
- FHIR conformance certification;
- superiority over MedAgentBench/FHIR-AgentBench agents.

Those require P08 matched evidence and appropriate clinical/standards validation.

## Exit gate

P07 closes only after:

1. SG-000010 scope and standards/source freezes are canonical;
2. renderer, representation controls, read-only actions, decision-case conversion, and local-export adapters are tested;
3. dataset/source registries pass validation;
4. final-test rejection remains active;
5. exact-head CI passes Ruff, mypy strict, pytest, and compileall on Linux/Windows × Python 3.11/3.12;
6. implementation merges with an expected-head guard;
7. post-main CI succeeds;
8. a separate canonical closeout marks SG-000010 `PROVEN` and advances to P08.
