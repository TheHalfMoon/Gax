# GAX Roadmap

The roadmap is gate-based. Percent-complete estimates are secondary to proven evidence.

## P00 — Research foundation

Goal: freeze the research contract before implementation.

Deliverables:

- mission and non-goals;
- literature map;
- architecture search space;
- GAXBench specification;
- data governance;
- reproducibility/claim policy;
- paper plan;
- SpecGrain frontier.

Exit gate: foundation PR reviewed and merged with no unresolved scope contradictions.

## P01 — Benchmark kernel

Build:

- benchmark item schema;
- metric library;
- risk-coverage evaluation;
- calibration metrics;
- provenance registry;
- deterministic synthetic fixtures;
- split/duplicate audit tools.

Exit gate: tiny public fixture reproduces identical metrics on Linux and Windows; metric tests include pathological cases.

## P02 — Baseline harness

Integrate reproducible adapters for:

- plain clinical encoder;
- GAX architecture candidates;
- CLM where feasible;
- Laya;
- decider;
- restricted-logit/open one-pass baseline;
- strong open structured-output LLM baseline.

Exit gate: one command runs matched benchmark slices and emits evidence packets.

## P03 — GAX v0 model

Implement the lowest-complexity clinical typed-decision model.

Required:

- action distributions;
- deterministic inference;
- checkpoint format;
- training script;
- model card draft;
- tiny overfit and smoke tests.

Exit gate: reproducible baseline model and no hidden generation.

## P04 — ECAL experiments

Test:

- state/action contrastive training;
- hard negatives;
- evidence objective;
- proper-scoring/calibration objective;
- replay.

Exit gate: keep only components with validated benefit.

## P05 — Native abstention

Build and compare:

- confidence/entropy threshold;
- learned sufficiency;
- calibrated selective policy;
- optional conformal/selective baseline.

Exit gate: selective model beats confidence-only control at matched coverage on frozen dev/validation tasks.

## P06 — Counterfactual and evidence program

Build controlled medical counterfactual and evidence intervention suites.

Exit gate:

- material edits change beliefs in expected direction;
- irrelevant edits remain stable within declared tolerance;
- evidence swap/contradiction results are interpretable.

## P07 — FHIR decision layer

Implement:

- FHIR canonicalizer;
- resource/tool action schemas;
- MedAgentBench/FHIR-AgentBench adapters where terms permit;
- representation controls.

Exit gate: reproducible interoperable benchmark table and no serialization-only artifact driving the headline result.

## P08 — Full evaluation

Freeze:

- models;
- benchmark;
- baselines;
- calibration;
- hardware protocol.

Run:

- primary experiments;
- ablations;
- distribution shift;
- failure analysis;
- efficiency;
- confidence intervals.

Exit gate: all paper figures/tables generated from evidence packets.

## P09 — Paper and public release

Prepare:

- manuscript;
- appendix;
- arXiv source;
- code release;
- model weights where license permits;
- GAXBench release/builders;
- fine-tuning notebooks;
- Hugging Face model/dataset/Space;
- citation metadata;
- release tag.

Exit gate: independent reproduction checklist passed and all public claims trace to evidence.

## P10 — Peer review / external validation

After arXiv:

- respond to public issues;
- invite independent reproduction;
- submit to an appropriate peer-reviewed venue based on active CFP and final contribution;
- add external datasets/clinical review only as genuinely new evidence;
- prepare journal extension only if it contains substantial new work.

## Compute rule

No founder-paid cloud compute.

Use free local/Colab/Kaggle resources for smoke/tutorial work and donated/sponsored/institutional compute for experiments that exceed those limits. Every research artifact records compute provenance.

## Release rule

Do not launch a model merely because training finished.

Release requires benchmark evidence, calibration/selective analysis, failure cases, model card, data/license review, and reproducibility artifacts.
