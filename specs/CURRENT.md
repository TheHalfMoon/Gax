# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN
- **SG-000004 — GAX-P02 External transport adapters and upstream revision freeze** — PROVEN

P02 external-adapter evidence:

- implementation PR: #10
- exact head: `955ff768a78a274aa5b1e609f6c49f58d8c26165`
- exact-head CI: run `36034345889` — SUCCESS
- exact-head tests: 50 passed on Linux/Windows, Python 3.11/3.12
- merge: `a2bc312a96d8df6413678038436a49a8c09e622a`
- post-merge CI: run `36034544718` — SUCCESS

State: **P02 ACTIVE**

Active frontier:

**P02 — Real Baseline Qualification and Remaining Controls**

Canonical infrastructure now includes:

- strict GAXBench evaluator and evidence packets;
- matched baseline runner;
- TypeSafe-compatible HTTP transport for CLM/Laya/decider-class servers;
- shell-free JSON-command transport;
- no-label-leak model rendering;
- frozen public source/license provenance for CLM, Laya, decider, and open-alternative-jev.

Remaining P02 work:

- select, freeze, and qualify a clinical encoder baseline;
- execute CLM with immutable model/tokenizer revisions on zero-founder-cost compatible compute;
- execute Laya with immutable model/tokenizer revisions on zero-founder-cost compatible compute;
- execute decider with immutable model/tokenizer revisions on zero-founder-cost compatible compute;
- qualify a restricted-logit/open one-pass checkpoint through the canonical transport;
- select, freeze, and qualify an open structured-output LLM control.

**Adapter qualification is not real-model qualification.** No external model result may enter a paper table until its immutable checkpoint, tokenizer, command, hardware, failures, and raw predictions are bound by an evidence packet.

Jev remains optional and must stay explicitly blocked unless reproducible access and terms permit matched measurement.

GAX v0 model training remains out of scope until the full P02 task set is canonically closed.

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
