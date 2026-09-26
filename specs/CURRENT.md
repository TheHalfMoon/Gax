# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN
- **SG-000004 — GAX-P02 External transport adapters and upstream revision freeze** — PROVEN
- **SG-000005 — GAX-P02 Phase closeout and final baseline identity freeze** — PROVEN
- **SG-000006 — GAX-P03 First trainable non-generative GAX v0** — PROVEN
- **SG-000007 — GAX-P04 ECAL controlled ablation framework** — PROVEN
- **SG-000008 — GAX-P05 Native information sufficiency and selective abstention** — PROVEN
- **SG-000009 — GAX-P06 Evidence interventions and counterfactual robustness** — PROVEN
- **SG-000010 — GAX-P07 FHIR interoperable read-only decision layer** — PROVEN
- **SG-000011 — GAX-P08 Final-test freeze and claim-evidence contract** — PROVEN

## Canonical P03 evidence chain

- implementation PR: #15
- implementation exact head: `18bad9e76ef161836875282e26acbaa0ff873a92`
- implementation exact-head CI: run `36239680443` — SUCCESS
- implementation merge: `5b51faf169ec828888d0f314acf2e8e8196bb5cb`
- implementation post-merge CI: run `36239756362` — SUCCESS
- model-visible-boundary correction PR: #16
- correction exact head: `89f19898ca4a46dcd9551979f216dc77737ef266`
- correction exact-head CI: run `36240225652` — SUCCESS
- corrected main: `d04cc0c63d83b944c2f1f3527a358848c27dd304`
- correction post-merge CI: run `36240311386` — SUCCESS
- final P03 feature revision: `sha256-word-v0.2`

P03 is **CLOSED_CANONICAL**. GAX has a genuine trainable non-generative typed-decision reference model with deterministic training, integrity-bound checkpointing, CLI, and direct GAXBench integration. Closeout review caught and corrected a model-visible evidence-label leak before freeze. Abstract smoke results remain infrastructure evidence only and are not clinical or paper performance results.

## Canonical P04 evidence chain

- implementation PR: #19
- implementation exact head: `d1f3315c949ed4a10783a6bfd3881e80521f8e74`
- implementation exact-head CI: run `36247918779` — SUCCESS
- implementation merge: `4baf5104314d1a1eecbb91cb7984ccb5e7df7866`
- implementation post-merge CI: run `36248363024` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- implementation qualification includes multi-positive alignment, deterministic hard-negative control, hidden-label-safe evidence intervention, Brier-gradient verification, equal-step replay/retention, explicit experiment context, and a governed deterministic fallback for exact zero-norm signed-hash cancellation
- all ECAL paper decisions remain `defer-real-data`; synthetic mechanism fixtures are implementation evidence only

P04 is **CLOSED_CANONICAL**. GAX has a qualified component-wise ECAL experiment framework, not an ECAL superiority claim. No ECAL component may enter the paper as a positive contribution until its preregistered keep/reject gate is evaluated on licensed, leakage-audited development evidence.

## Canonical P05 evidence chain

- implementation PR: #22
- implementation exact head: `309e7c108f4dab87f80d2ab24849d9c3032102db`
- implementation exact-head CI: run `36251925614` — SUCCESS
- implementation merge: `ced14e8e8f599f8022a273c762fda56452c335c6`
- implementation post-merge CI: run `36252259764` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- confidence-only controls: max probability, normalized entropy confidence, top-1/top-2 margin
- learned information sufficiency remains a separate model-visible probability-like signal trained without updating the frozen action model
- calibration-split threshold selection remains isolated from validation evaluation; P05 final-test selector evaluation is rejected
- unsafe-commit and over-abstention accounting are both required under matched coverage
- learned sufficiency paper decision remains `defer-real-data`; synthetic fixtures are implementation evidence only

P05 is **CLOSED_CANONICAL** as an implementation and selective-evaluation framework result. It does not establish clinical safety, a formal conformal guarantee, or learned-sufficiency superiority. A paper-level keep decision still requires licensed, leakage-audited development evidence under the preregistered matched-coverage gate.

## Canonical P06 evidence chain

- research contract: Issue #23
- implementation PR: #25
- implementation exact head: `e848499a2b1500de6fda2e1f55afd5afc79beae2`
- implementation exact-head CI: run `36254832102` — SUCCESS
- implementation merge: `a63704e89d4ad81c97358d3f65f5deb18cdcc041`
- implementation post-merge CI: run `36254912534` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- paired evaluator freezes total-variation and natural-log Jensen-Shannon conventions, same-split/source/group lineage, exact denominator alignment, explicit abstention, directional action/support/sufficiency response, and irrelevant-edit stability
- P06 CLI rejects final-test items and binds runs to a 40-character git revision plus compute provenance and artifact hashes
- abstract fixtures contain no patient data, PHI, real medical thresholds, diagnosis rules, treatment guidance, or clinical source claims
- all P06 paper mechanism decisions remain `defer-real-data`

P06 is **CLOSED_CANONICAL** as a paired intervention/evaluation framework result. It does not establish clinical safety, causal identification, or medical counterfactual superiority. The proven surface is a governed pair manifest/evaluator and synthetic mechanics suite; real medical intervention construction and paper-level keep decisions require licensed, reviewed, leakage-audited development evidence.

## Canonical P07 evidence chain

- research contract: Issue #27 / SG-000010
- implementation PR: #28
- implementation exact head: `01e818a5139dea3d0ea3258b49f60f24593ade41`
- implementation exact-head CI: run `36258059106` — SUCCESS
- implementation merge: `a9128a7f8de49dfa614ded1bd472cf1230f0041a`
- implementation post-merge CI: run `36258143003` — SUCCESS
- closeout PR: #29
- closeout exact head: `ee04a3ae2e5c3c82aa82324ae0aaf566ed9dd996`
- closeout exact-head CI: run `36258444879` — SUCCESS
- canonical closeout merge: `250ffdf6d3c91c030c79fc2835daabe605efef40`
- closeout post-main CI: run `36258521894` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- canonical published FHIR reference is R5 5.0.0 while R4 benchmark inputs preserve explicit source-native identity
- P07 makes no R4-to-R5 semantic conversion claim and does not treat deterministic rendering as HL7 profile/conformance certification
- FHIR actions are read-only; POST/PUT/PATCH/DELETE templates are rejected and no model-selected EHR write operation is exposed
- MedAgentBench and FHIR-AgentBench repository identities/revisions are frozen while restricted external data/runtime artifacts remain separately governed
- public P07 fixtures are abstract CC0 synthetic content and final-test conversion remains blocked during P07

P07 is **CLOSED_CANONICAL** as an interoperability and evaluation-contract result. It proves deterministic FHIR rendering, explicit version identity, read-only action boundaries, representation controls, and frozen local-export adapter protocols. It does not prove FHIR benchmark superiority, clinical correctness, FHIR conformance certification, or semantic R4-to-R5 equivalence.

## Canonical P08 pre-test freeze evidence chain

- research contract: Issue #30 / SG-000011
- implementation PR: #31
- implementation exact head: `45bd2dcdcd1e4b96be20ffbf0c77080d3bd690d0`
- implementation exact-head CI: run `36259370327` — SUCCESS
- implementation merge: `1818aa67f93d5ed484611392011934ecfa770a1c`
- implementation post-main CI: run `36259485349` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- final-test access is sealed by default and authorization requires a clean tree, frozen benchmark manifests, audit hashes, frozen system identities, calibration/statistics/protocol contracts, and a digest-bound separate authorization artifact
- the P08 gate is hard-bound to the canonical P07 closeout merge `250ffdf6d3c91c030c79fc2835daabe605efef40` and post-main run `36258521894`
- claim states are machine-readable; affirmative export is restricted to `supported` claims carrying evidence-packet identifiers
- null, rejected, blocked, and exploratory results remain visible by contract
- the 2026-09-26 literature refresh raises the novelty bar beyond generic typed-decision/Jev-like behavior

SG-000011 is **CLOSED_CANONICAL** as a pre-test governance result. It proves the freeze/authorization/claim discipline only. It does not authorize final-test access and does not establish any model, mechanism, clinical, FHIR, calibration, or efficiency result.

Active frontier:

**P08 — Full Paper Evaluation / statistical and evidence infrastructure**

P08 remains open. The next governed unit builds the paired uncertainty and evidence aggregation layer needed before a real final-test freeze:

- paired bootstrap confidence intervals with frozen seeds/replicate counts;
- deterministic evidence AUROC/AUPRC with tie handling and undefined-case reporting;
- reliability-bin artifacts alongside ECE/NLL/Brier;
- failure-preserving aggregation that never silently changes denominators;
- table/figure source manifests binding every derived artifact to raw run/evidence identifiers;
- deterministic primary-comparison report serialization.

Core P08 rules remain:

```text
final-test access != model selection
confidence != information sufficiency
FHIR formatting != clinical correctness
faster on different hardware != speed superiority
missing/failed inference != silent exclusion
negative result != disposable result
```

Final-test labels remain sealed. No paper-level claim for ECAL, learned information sufficiency, evidence grounding, counterfactual robustness, FHIR gains, efficiency, or baseline superiority is authorized until the real P08 data/model/protocol freeze is populated, audited, digest-authorized, and evaluated under the preregistered contract. Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
