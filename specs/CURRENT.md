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

P04 is **CLOSED_CANONICAL**. GAX has a qualified component-wise ECAL experiment framework, not an ECAL superiority claim. No ECAL component may enter the paper as a positive contribution until its preregistered keep/reject gate is evaluated on licensed, leakage-audited development evidence.

Canonical P05 evidence chain:

- implementation PR: #22
- implementation exact head: `309e7c108f4dab87f80d2ab24849d9c3032102db`
- implementation exact-head CI: run `36251925614` — SUCCESS
- implementation merge: `ced14e8e8f599f8022a273c762fda56452c335c6`
- implementation post-merge CI: run `36252259764` — SUCCESS
- exact-head and post-main matrices: Linux/Windows × Python 3.11/3.12 — SUCCESS
- Windows Python 3.11 exact-head qualification: Ruff PASS, mypy strict PASS across 17 source files, 84 pytest tests PASS, compileall PASS
- confidence-only controls: max probability, normalized entropy confidence, top-1/top-2 margin
- learned information sufficiency remains a separate model-visible probability-like signal trained without updating the frozen action model
- calibration-split threshold selection remains isolated from validation evaluation; P05 final-test selector evaluation is rejected
- unsafe-commit and over-abstention accounting are both required under matched coverage
- learned sufficiency paper decision remains `defer-real-data`; synthetic fixtures are implementation evidence only

P05 is **CLOSED_CANONICAL** as an implementation and selective-evaluation framework result. It does not establish clinical safety, a formal conformal guarantee, or learned-sufficiency superiority. A paper-level keep decision still requires licensed, leakage-audited development evidence under the preregistered matched-coverage gate.

Active frontier:

**P06 — Evidence Interventions and Counterfactual Clinical Robustness**

P06 tests whether GAX changes action beliefs for materially relevant state/evidence changes while remaining stable to irrelevant edits. It is a falsification program for grounding and counterfactual claims, not a generic data-augmentation phase.

Core research invariants:

```text
uses evidence != merely receives evidence
material sensitivity != irrelevant-edit instability
counterfactual robustness != causal identification != clinical safety
```

Every base/intervention family must remain in the source split. Final-test labels cannot be used to generate interventions, tune thresholds, choose mechanisms, or repair failures. Evidence relation labels remain benchmark supervision and may not become model-visible inference features unless a future explicitly governed task changes that contract.

Synthetic P06 fixtures may prove parser, lineage, metric, and invariance behavior only. Real medical counterfactuals or evidence-based paper claims require licensed sources, leakage controls, explicit review, matched baselines, and artifact-backed statistical evidence.

P06 research contract: Issue #23.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
