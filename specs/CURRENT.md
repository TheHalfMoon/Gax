# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN
- **SG-000004 — GAX-P02 External transport adapters and upstream revision freeze** — PROVEN
- **SG-000005 — GAX-P02 Phase closeout and final baseline identity freeze** — PROVEN

Canonical P02 closeout evidence:

- phase-closeout PR: #12
- exact head: `d413bd845b1cff66a90ae1da3c5ea07b42431216`
- exact-head CI: run `36035557131` — SUCCESS
- phase-closeout merge: `718badb174c984c1dfb7571b4246a9cfb5fd274c`
- post-merge CI: run `36035720538` — SUCCESS

P02 is **CLOSED_CANONICAL**. The baseline comparison infrastructure is frozen sufficiently to begin GAX itself. Real external-model benchmark execution remains a P08 responsibility after benchmark, model, calibration, and hardware protocol freeze.

Active frontier:

**P03 — GAX v0**

P03 may implement the lowest-complexity GAX typed-decision model, training pipeline, checkpoint contract, deterministic inference path, synthetic smoke training, and model-card draft.

P03 must not introduce a headline clinical benchmark claim, real-patient data, PHI, or autonomous clinical-use claim. The purpose of P03 is to prove that a genuine trainable non-generative GAX model exists and is reproducible before ECAL research begins.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
