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

Canonical P03 evidence chain:

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

Canonical P04 evidence chain:

- implementation PR: #19
- implementation exact head: `d1f3315c949ed4a10783a6bfd3881e80521f8e74`
- implementation exact-head CI: run `36247918779` — SUCCESS
- implementation merge: `4baf5104314d1a1eecbb91cb7984ccb5e7df7866`
- implementation post-merge CI: run `36248363024` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- implementation qualification includes multi-positive alignment, deterministic hard-negative control, hidden-label-safe evidence intervention, Brier-gradient verification, equal-step replay/retention, explicit experiment context, and a governed deterministic fallback for exact zero-norm signed-hash cancellation
- all ECAL paper decisions remain `defer-real-data`; synthetic mechanism fixtures are implementation evidence only

P04 is **CLOSED_CANONICAL**. GAX now has a qualified component-wise ECAL experiment framework, not an ECAL superiority claim. No ECAL component may enter the paper as a positive contribution until its preregistered keep/reject gate is evaluated on licensed, leakage-audited development evidence.

Active frontier:

**P05 — Native Information Sufficiency and Selective Abstention**

P05 separates action preference from permission to act. It compares max-probability, entropy, and top-1/top-2 margin controls against a dedicated learned information-sufficiency signal and a calibration/selective-risk control under matched coverage.

Core research invariant:

```text
action confidence != information sufficiency
```

A P05 selector may use model-visible state/candidate information and action-distribution summaries, but it may not consume `gold.sufficient`, gold actions, provenance/split labels, benchmark-only evidence relations, or final-test annotations at inference. Thresholds and calibration parameters must be selected without final-test labels.

P05 abstention means **do not commit to a clinical action from the current information**. Later P06/P07 work may map insufficiency to evidence retrieval, clarification, FHIR queries, or escalation; those behaviors remain outside P05.

Primary selective comparisons use matched coverage and report both unsafe commits and over-abstention. A selector cannot win merely by refusing nearly everything.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
