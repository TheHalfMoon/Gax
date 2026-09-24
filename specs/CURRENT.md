# Current Frontier

Program: **GAX**

Completed grains:

- **SG-000001 — GAX-P00 Research foundation and publication contract** — PROVEN
- **SG-000002 — GAX-P01 Benchmark kernel and selective-risk metric contract** — PROVEN
- **SG-000003 — GAX-P02 Core matched baseline harness and evidence-packet contract** — PROVEN
- **SG-000004 — GAX-P02 External transport adapters and upstream revision freeze** — PROVEN

Canonical SG-000004 evidence:

- implementation PR: #10
- exact head: `955ff768a78a274aa5b1e609f6c49f58d8c26165`
- exact-head CI: run `36034345889` — SUCCESS
- exact-head tests: 50 passed on Linux/Windows, Python 3.11/3.12
- implementation merge: `a2bc312a96d8df6413678038436a49a8c09e622a`
- implementation post-merge CI: run `36034544718` — SUCCESS
- SG-000004 closeout merge: `b6b30af27c3d6e9d603c5b4407b7ba1e61ab6d24`
- closeout post-main CI: run `36034993157` — SUCCESS

State: **P02 CLOSEOUT CANDIDATE**

Active grain:

**SG-000005 — GAX-P02 Phase closeout and final baseline identity freeze**

P02 now has the infrastructure required by Issue #6:

- strict GAXBench evaluator and evidence packets;
- matched baseline runner;
- TypeSafe-compatible HTTP transport;
- shell-free JSON-command transport;
- no-label-leak model rendering;
- explicit failure accounting;
- frozen source/license records for CLM, Laya, decider, and restricted-logit;
- frozen clinical encoder control: `thomas-sounack/BioClinical-ModernBERT-base@5e17e2f25260b6993e0fb60485f94678ff29779a` (MIT);
- frozen structured-output control: `Qwen/Qwen3.5-4B@a7b0d22b993d71000cf2eadfb37222a67cee521e` (Apache-2.0).

Real external-model execution is a **P08 frozen-evaluation responsibility**, not a prerequisite for P03. No external benchmark score has been introduced in P02.

Jev remains explicitly blocked unless reproducible access and terms permit matched measurement.

If SG-000005 qualifies and closes canonically, the next frontier is:

**P03 — GAX v0**

Research claims remain subject to `docs/research/REPRODUCIBILITY.md`.
