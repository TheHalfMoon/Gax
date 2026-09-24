# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN

P02 core harness evidence:

- implementation PR: #8
- exact head: `5e90f5ebad930d3467fb550c8f707c689e594cab`
- exact-head CI: run `36030392516` — SUCCESS
- merge: `31ca7827d230fa2cc9c276b0f0a38e9118b3f4eb`
- post-merge CI: run `36030695502` — SUCCESS

State: **P02 ACTIVE**

Active frontier:

**P02 — External Baseline Adapters and Qualification**

Active implementation grain: **SG-000004 — External transport adapters and upstream revision freeze**.

The core adapter/runner/evidence-packet machinery is canonical. The remaining P02 work is to implement and qualify the required external baseline classes without changing benchmark semantics:

- clinical encoder;
- CLM;
- Laya;
- decider;
- restricted-logit/open one-pass control;
- open structured-output LLM.

Jev remains optional and must stay explicitly blocked unless reproducible access and terms permit matched measurement.

GAX v0 model training remains out of scope until the full P02 task set is canonically closed.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
