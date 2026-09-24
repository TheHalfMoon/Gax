# GAX Paper Plan

Working title:

**GAX: Evidence-Calibrated System-One Models for Clinical Decision Making**

Working subtitle:

**Fast Typed Decisions with Grounded Uncertainty, Abstention, and FHIR-Aware Action Selection**

The title is provisional. The final title must describe demonstrated results, not aspirations.

## Core paper story

Current medical AI is dominated by autoregressive text generation. Many downstream workflows, however, require a bounded decision: choose an action, score a risk, verify support, or decide that the available information is insufficient.

GAX tests whether a dedicated decision model can provide a better interface and a better efficiency/calibration/selectivity trade-off for those workflows.

The central distinction is:

```text
highest-probability action != permission to act
```

The model may have a top-ranked action while still reporting insufficient evidence and abstaining.

## Claimed contributions — gated

The manuscript may claim only contributions that pass their gates.

### C1. Health-specific System-One decision model

Gate:

- reproducible open training/inference implementation;
- meaningful performance against both general typed-decision and health-model baselines;
- not merely a wrapper around a generative API.

### C2. Evidence-Calibrated Action Learning

Gate:

- full objective and data construction released;
- ablation demonstrates value beyond action-only training;
- benefit survives held-out source/task evaluation.

If the ablation fails, ECAL is removed or reframed as a negative result.

### C3. Native selective abstention

Gate:

- lower risk at matched coverage than confidence-threshold baselines;
- explicit missing/contradictory information evaluation;
- no trivial gain through excessive abstention.

### C4. Evidence grounding

Gate:

- evidence-swap and contradiction tests;
- support discrimination metrics;
- lower unsupported-confident-action rate.

### C5. FHIR-aware decision layer

Gate:

- interoperable benchmark evaluation;
- comparison against reasonable retrieval/routing baselines;
- serialization-control experiments.

### C6. GAXBench

Gate:

- released construction/evaluation code;
- license-compatible data artifacts or builders;
- contamination and split audits;
- multiple model families evaluated.

## Paper experiments

### E0 — baseline freeze

Freeze model versions, dataset revisions, prompts/interfaces, hardware classes, and evaluation code.

### E1 — architecture comparison

Compare:

- plain clinical encoder classifier;
- single-encoder typed heads;
- contrastive state/action model;
- one-pass decoder/answer-slot model.

### E2 — domain backbone

Clinical vs general encoder at matched size/training.

### E3 — evidence

Action-only vs evidence-supervised vs evidence+counterfactual.

### E4 — abstention

Confidence threshold vs learned sufficiency vs calibrated selective policy.

### E5 — counterfactuals

No CF training vs CF training; material sensitivity and irrelevant stability.

### E6 — FHIR

Representation and action-routing comparisons.

### E7 — efficiency

Matched hardware where possible; otherwise strictly separated hardware tables with no direct speed superiority claim.

### E8 — scale

GAX-base vs GAX-large if compute becomes available without founder cloud spend.

## Required ablation table

Rows:

- full GAX;
- minus evidence objective;
- minus abstention objective;
- minus counterfactual objective;
- minus calibration/proper-scoring component;
- action-only;
- backbone-only baseline.

Columns:

- action accuracy;
- NLL/Brier;
- ECE;
- risk@80% coverage;
- AURC;
- evidence AUPRC;
- counterfactual sensitivity/stability;
- latency.

## Required figures

1. **Architecture figure** — state/actions/evidence -> typed beliefs + sufficiency -> policy.
2. **Risk-coverage curves** — central safety figure.
3. **Calibration reliability diagram** — in-domain and shifted.
4. **Evidence intervention figure** — support/irrelevant/contradictory evidence.
5. **Counterfactual response figure** — material vs irrelevant edits.
6. **Pareto frontier** — decision quality vs latency/memory.
7. **FHIR workflow figure** — where GAX sits in an agent loop.

Every plot must be regenerated from versioned raw result files.

## Required tables

- main benchmark table;
- abstention/selective prediction table;
- evidence/counterfactual table;
- FHIR table;
- ablation table;
- efficiency table;
- data/source/size table;
- failure taxonomy.

## Failure analysis

The paper must include examples of:

- confident wrong actions;
- correct action but wrong sufficiency;
- over-abstention;
- evidence-following failures;
- counterfactual insensitivity;
- irrelevant perturbation instability;
- FHIR retrieval/representation errors;
- specialty/source shifts.

Examples must not expose restricted clinical data.

## Reproducibility appendix

Include:

- exact repository tag/commit;
- model and dataset immutable revisions;
- training commands;
- evaluation commands;
- seed table;
- hardware;
- software lock;
- compute accounting;
- hyperparameter search budget;
- exclusion/failure policy;
- confidence interval method.

## Publication path

### Stage A — arXiv

Release only after the central claims are backed by the frozen evaluation package.

Candidate categories are chosen at submission time based on the final paper scope; likely computer-science ML/NLP categories with appropriate health framing.

### Stage B — peer review

Choose based on the demonstrated contribution and active calls. Candidate communities:

- Machine Learning for Healthcare / health-ML venues;
- CHIL-style health inference/learning venues;
- major ML venues if the methodological contribution is broad;
- major NLP venues if the evidence/clinical-language contribution is dominant.

No deadline or venue acceptance is assumed before the corresponding call is verified.

### Stage C — journal extension

Only if there is genuinely new validation beyond the conference/preprint, potentially including broader clinical review, external datasets, or prospective-style evaluation.

## Claim language

Forbidden without direct evidence:

- "safe for clinical use";
- "cannot hallucinate";
- "best medical AI";
- "state of the art" based on unmatched protocols;
- "clinically validated" without appropriate clinical validation.

Preferred:

- measured task-specific statements with dataset version, population/task scope, uncertainty, and limitations.

## Release package

The arXiv release should coincide with, or be followed promptly by:

- source code;
- weights if license permits;
- model card;
- GAXBench code/data builders;
- benchmark card;
- fine-tuning notebook;
- reproducibility manifest;
- raw aggregate-safe results;
- paper source;
- citation metadata;
- demo clearly marked as research use.
