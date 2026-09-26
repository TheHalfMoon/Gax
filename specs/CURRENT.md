# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN
- **SG-000004 — GAX-P02 External transport adapters and upstream revision freeze** — PROVEN
- **SG-000005 — GAX-P02 Phase closeout and final baseline identity freeze** — PROVEN
- **SG-000006 — GAX-P03 First trainable non-generative GAX v0** — PROVEN

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
- final feature revision: `sha256-word-v0.2`

P03 is **CLOSED_CANONICAL**. GAX has a genuine trainable non-generative typed-decision reference model with deterministic training, integrity-bound checkpointing, CLI, and direct GAXBench integration. Closeout review caught and corrected a model-visible evidence-label leak before freeze. Abstract smoke results remain infrastructure evidence only and are not clinical or paper performance results.

Active frontier:

**P04 — ECAL Research**

P04 evaluates Evidence-Calibrated Action Learning as a set of falsifiable components rather than a predeclared successful objective. Candidate mechanisms include bidirectional/multi-positive state-action contrastive structure, hard-negative ranking, evidence supervision, proper-scoring/calibration objectives, and replay/retention.

Every component must be independently switchable and tested under a matched ablation. Components that fail their preregistered development gate are rejected or reframed rather than retained for a stronger story. P04 must not tune on final test labels or convert synthetic mechanistic results into clinical claims.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
