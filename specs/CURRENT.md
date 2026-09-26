# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN
- **SG-000004 — GAX-P02 External transport adapters and upstream revision freeze** — PROVEN
- **SG-000005 — GAX-P02 Phase closeout and final baseline identity freeze** — PROVEN
- **SG-000006 — GAX-P03 First trainable non-generative GAX v0** — PROVEN

Canonical P03 closeout evidence:

- implementation PR: #15
- exact head: `18bad9e76ef161836875282e26acbaa0ff873a92`
- exact-head CI: run `36239680443` — SUCCESS
- implementation merge: `5b51faf169ec828888d0f314acf2e8e8196bb5cb`
- post-merge CI: run `36239756362` — SUCCESS

P03 is **CLOSED_CANONICAL**. GAX now has a genuine trainable non-generative typed-decision reference model with deterministic training, integrity-bound checkpointing, CLI, and direct GAXBench integration. Its abstract synthetic smoke results are infrastructure evidence only and are not clinical or paper performance results.

Active frontier:

**P04 — ECAL Research**

P04 may implement and test the proposed Evidence-Calibrated Action Learning components under controlled ablations: state/action contrastive structure, hard negatives, evidence supervision, proper-scoring/calibration objectives, and replay/retention.

P04 must treat every ECAL component as an unproven hypothesis. A component is retained only if a preregistered matched experiment supports its claimed effect. P04 must not tune on final test labels or convert synthetic smoke gains into clinical claims.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
