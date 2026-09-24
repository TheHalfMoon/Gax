# GAXBench Specification

GAXBench evaluates **decision quality under uncertainty**, not conversational quality.

## Design principles

1. Health decisions are evaluated as explicit action sets.
2. Abstention is evaluated at matched coverage, not rewarded in isolation.
3. Evidence grounding is tested through interventions on evidence.
4. Counterfactual sensitivity and irrelevant-edit stability are both required.
5. FHIR tasks measure interoperable data handling rather than prompt memorization.
6. Every item carries provenance and split lineage.
7. No aggregate score may hide catastrophic performance on a safety-relevant slice.

## Suite

### GAXBench-Knowledge

Purpose: establish ordinary closed-action medical knowledge competence.

Candidate public sources are admitted only after license and contamination review. This slice is not sufficient for a clinical-safety claim.

Metrics: accuracy, macro-F1 where appropriate, NLL, Brier, ECE.

### GAXBench-Abstain

Purpose: test whether the model refuses to commit when information is insufficient.

Item families:

- gold answer/action removed from the candidate set;
- critical state field masked;
- question/state partially hidden;
- mutually insufficient evidence;
- out-of-scope specialty/task;
- impossible or internally inconsistent request.

External benchmark alignment should include MedQAbstain where licensing and task conversion permit.

Metrics:

- risk-coverage curve;
- AURC;
- risk@50/80/90% coverage;
- abstention precision/recall;
- unsafe-commit rate;
- calibration of sufficiency.

### GAXBench-Evidence

Purpose: determine whether decisions respond to evidence quality.

Each group can contain:

- supporting evidence;
- irrelevant but plausible evidence;
- contradictory evidence;
- evidence from a neighboring condition;
- evidence removed.

Metrics:

- support AUROC/AUPRC;
- action accuracy conditioned on evidence state;
- confidence delta under evidence swap;
- unsupported confident action rate.

### GAXBench-CF

Purpose: test clinically material counterfactual behavior.

Examples should change one controlled fact such as:

- lab value and unit;
- allergy presence;
- pregnancy status where clinically relevant;
- medication exposure;
- symptom duration;
- age band;
- vital-sign severity;
- renal/hepatic function indicator.

Every material edit is paired with irrelevant edits such as formatting, field order, non-material wording, or unrelated history.

Metrics:

- directional sensitivity;
- required action flip rate;
- irrelevant-edit agreement;
- probability stability;
- counterfactual calibration.

### GAXBench-FHIR

Purpose: evaluate decision layers in interoperable EHR contexts.

Use permitted/synthetic resources and established public benchmark infrastructure where licenses permit, including comparisons with MedAgentBench and FHIR-AgentBench task families.

Task classes:

- identify relevant resource type;
- choose next retrieval/action;
- route to appropriate tool;
- decide whether enough information has been gathered;
- verify a proposed action against retrieved evidence;
- abstain/escalate when required data are absent.

### GAXBench-Shift

Distribution-shift slices:

- source/institution where available;
- specialty;
- note style;
- synthetic vs natural language;
- coding vocabulary;
- long vs short context;
- demographic subgroup only when the source supports ethical, statistically meaningful analysis.

### GAXBench-Ops

Systems evaluation:

- p50/p95 latency;
- throughput;
- peak accelerator memory;
- CPU memory;
- model load time;
- cache hit/miss behavior;
- candidate-set scaling;
- context-length scaling;
- optional energy proxy with disclosed measurement method.

## Benchmark item schema

```json
{
  "id": "gaxb-...",
  "source_id": "...",
  "split": "test",
  "task_family": "abstain_missing_information",
  "state": {},
  "actions": [
    {"id": "a", "description": "..."}
  ],
  "gold": {
    "action": "a",
    "sufficient": false
  },
  "evidence": [
    {"id": "e1", "text": "...", "relation": "support"}
  ],
  "counterfactual_group": null,
  "provenance": {
    "dataset": "...",
    "revision": "...",
    "license": "...",
    "transform_revision": "..."
  }
}
```

The public schema must support label hiding for blind test evaluation.

## Split policy

- patient/entity-disjoint before transformation;
- near-duplicate filtering after transformation;
- task-family holdouts for generalization;
- benchmark test labels frozen before model selection;
- hidden test split preferred for public leaderboard-style evaluation;
- no teacher generation from held-out test labels.

## Statistical reporting

Primary comparisons use paired evaluation on identical items.

Required:

- 95% confidence intervals;
- bootstrap seed and replicate count;
- number of examples per slice;
- missing/failed inference count;
- exact exclusion rules;
- per-slice metrics.

Where a result is exploratory, it must be labeled exploratory.

## Benchmark governance

A benchmark release requires:

- license audit;
- dataset card;
- construction code;
- deterministic build where permitted;
- checksums;
- contamination notes;
- known limitations;
- versioned changelog.

Benchmark revisions that alter labels or membership increment a major/minor version and invalidate direct comparison unless migration is documented.
