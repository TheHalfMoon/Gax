# P08 Literature Refresh — 2026-09-26

This document is a time-stamped positioning refresh for the P08 freeze phase. It supplements `LITERATURE_MAP.md`; it does not replace a full pre-submission systematic search.

## 1. Why the novelty bar changed

The open System-One / typed-decision ecosystem expanded rapidly after the September 2026 Jev launch. Multiple independent projects now demonstrate some combination of one-pass typed decisions, option-logit readout, compact encoders, calibration, Jev-compatible APIs, and local inference.

Examples that must be considered during the final related-work freeze include:

- `Mapika/decider` — one-pass typed decisions with Qwen3.5-family models, calibration/selective metrics, serving and evaluation infrastructure;
- `kshetrajna12/reflex` — open typed-decision reconstruction using open-weight models;
- `malevrigns/agent-jev` — small agent-focused typed-decision model with reported calibration metrics;
- `mateolafalce/system-one-model` — ModernBERT-based local typed-decision training/calibration workflow;
- the broader community catalogs and surveys that appeared immediately after Jev.

Community directories and self-reported benchmark tables are useful for discovery but are **not peer-reviewed evidence** and must not be copied into GAX as scientific ground truth without independent reproduction.

### P08 implication

GAX must not claim novelty merely because it:

- avoids autoregressive text generation;
- returns a closed-set probability distribution;
- exposes a Jev-compatible typed API;
- uses a compact encoder;
- reports ECE/Brier/AURC;
- runs faster than a large generative model on unmatched hardware.

Those are now baseline/category properties, not a sufficient paper thesis.

## 2. Medical abstention is already an explicit benchmark topic

Cocchieri et al., **“LLMs (Almost) Never Abstain Under Medical Uncertainty”**, ACL 2026, introduces MedQAbstain.

Source:
https://aclanthology.org/2026.acl-long.1365/

The benchmark removes the gold answer from medical MCQA cases and introduces an explicit abstention option. The paper reports systematic overcommitment by state-of-the-art LLMs across its uncertainty settings.

### P08 implication

GAX cannot claim novelty merely for including an abstain option. The stronger question is whether **separate learned information sufficiency** improves risk at matched coverage relative to max-probability, entropy, and margin controls, especially under missing, contradictory, and out-of-scope information.

A GAX abstention claim therefore requires:

- matched-coverage evaluation;
- unsafe-commit and over-abstain accounting;
- calibration of the sufficiency signal;
- a clear distinction between the action distribution and permission to act.

## 3. FHIR evaluation now has a strong peer-reviewed reference

Lee et al., **FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering**, Machine Learning for Health 2026, PMLR 297.

Source:
https://proceedings.mlr.press/v297/lee26a.html

FHIR-AgentBench grounds 2,931 real-world clinical questions in HL7 FHIR and evaluates retrieval strategies, specialized tools, interaction patterns, and reasoning strategies.

### P08 implication

GAX should not reduce FHIR evaluation to serialization accuracy or a synthetic routing demo. Paper-level FHIR evidence should separate:

- resource/action routing;
- retrieval sufficiency and stopping;
- representation controls;
- unsupported-action rate;
- retrieval quality where valid gold labels exist;
- full generative-agent/QA success when externally evaluated.

A representation gain that disappears when JSON order/narrative/text controls change is not a FHIR-native capability claim.

## 4. MedAgentBench remains a realistic EHR-agent anchor

Jiang et al., **MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents**, NEJM AI 2025.

Sources:
https://arxiv.org/abs/2501.14654
https://github.com/stanfordmlgroup/MedAgentBench

The benchmark includes 300 physician-authored patient-specific tasks across 10 categories, 100 realistic patient profiles, and more than 700,000 record data elements in a FHIR-compliant interactive environment.

### P08 implication

GAX should distinguish the bounded decision layer from a full planning agent. If GAX improves tool/resource selection but not end-to-end agent success, the paper must report that boundary rather than implying full-agent superiority.

## 5. Evidence verification has a strong neighboring literature

Yun et al., **Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards**, EMNLP 2025.

Source:
https://aclanthology.org/2025.emnlp-main.837/

Med-PRM verifies intermediate reasoning steps against retrieved guidelines and literature and uses the resulting process rewards to improve medical reasoning systems.

### P08 implication

GAX is not a process-reasoning generator, so Med-PRM is not an architectural equivalent. However, it raises the evidence bar: a GAX grounding claim must demonstrate response to controlled evidence interventions, not merely produce an evidence-support score that correlates with the action score.

## 6. Strongest defensible GAX paper thesis after refresh

The most defensible current thesis is not “a medical Jev.” It is:

> A compact non-generative clinical decision layer can be evaluated as a probabilistic selective predictor whose action belief, evidence support, and information sufficiency are distinct signals, and whose behavior can be stress-tested through evidence interventions, clinically controlled counterfactuals, and FHIR-aware action selection.

The empirical paper becomes important only if P08 establishes useful trade-offs against strong open typed-decision, clinical-encoder, restricted-logit, and generative structured-output controls.

## 7. Candidate contribution ladder

The final manuscript should promote only contributions that survive their P08 gates.

### Contribution A — GAX decision model

Candidate claim: compact health-specialized typed decision model with a favorable accuracy/calibration/selectivity trade-off.

Required evidence: matched baselines, confidence intervals, multiple seeds where train-sensitive, no hidden-label leakage.

### Contribution B — Information sufficiency

Candidate claim: a separate learned sufficiency signal improves selective risk at matched coverage.

Required evidence: max-probability, entropy, and margin controls; unsafe-commit/over-abstain analysis; sufficiency calibration.

### Contribution C — Evidence-aware learning

Candidate claim: evidence-aware training improves behavior under support/removal/swap/contradiction interventions.

Required evidence: paired intervention design, correct directional probability change, irrelevant-control stability, no evidence-label leakage.

### Contribution D — Counterfactual robustness

Candidate claim: clinically material state changes affect decisions while irrelevant edits remain stable.

Required evidence: reviewed transformation validity and matched pair lineage. Do not use causal language unless the identification design justifies it.

### Contribution E — FHIR-aware decision layer

Candidate claim: GAX provides useful bounded resource/action selection or stopping decisions in interoperable EHR workflows.

Required evidence: real source-qualified benchmark evaluation, representation controls, version identity, failure accounting, and comparison with appropriate agent/generative controls.

### Contribution F — Efficiency

Candidate claim: GAX occupies a favorable quality/latency/memory Pareto point.

Required evidence: matched hardware and frozen systems protocol. Cross-hardware absolute results may be reported but cannot establish direct speed superiority.

## 8. Claims that should not appear without new evidence

Do not write any of the following as affirmative claims unless a future dedicated validation supports them:

- “clinically safe”;
- “cannot hallucinate”;
- “causally grounded”;
- “FHIR compliant/certified” based only on deterministic rendering;
- “better than Jev” without reproducible matched access;
- “state of the art” based on incomparable benchmark variants;
- “faster” based on different hardware or caching protocols;
- “knows when not to act” as an empirical statement until selective-risk gates pass.

The tagline may remain project motivation, but the paper abstract must distinguish motivation from demonstrated result.

## 9. Next refresh

Repeat this search immediately before manuscript claim freeze and again immediately before peer-reviewed submission. Record new concurrent work in the claim ledger and narrow GAX claims when overlap appears.
