# GAX Architecture Search

This document defines the architecture search space and invariants. It does not declare a winner before experiments.

## Invariants

Every GAX implementation must:

- accept an explicit finite action set;
- return normalized probabilities over that action set;
- expose abstention/sufficiency separately from top-1 action confidence;
- preserve stable machine-readable output;
- support deterministic evaluation mode;
- record model, tokenizer, calibration, and policy revisions in responses or traces;
- never convert hidden free-form generation into a typed answer and call that "non-generative".

## Input representation

Canonical input has three independently identifiable channels:

1. **state** — clinical narrative, structured record, or canonicalized FHIR-derived context;
2. **actions** — caller-provided allowed actions with optional descriptions;
3. **evidence** — optional support passages/facts with provenance IDs.

Structured data is canonicalized with stable ordering and explicit missingness. FHIR resources must preserve resource type, identifiers local to the fixture, coding system, code, value, unit, status, and clinically material timestamps where needed.

## Architecture A — single encoder with typed heads

A clinical bidirectional encoder produces state representations. Candidate actions are encoded through the same backbone or an action tower. Heads produce:

- action logits;
- evidence-support score;
- sufficiency score;
- optional ordinal/risk distributions.

This is the lowest-complexity candidate and establishes whether a domain encoder alone captures most gains.

## Architecture B — contrastive state/action model

State and actions are embedded separately:

```text
z_s = normalize(P_s(E(state, evidence)))
z_a = normalize(P_a(E(action)))
logit(a_i | s) = exp(tau) * dot(z_s, z_a_i)
```

Advantages:

- reusable action embeddings;
- efficient large candidate sets;
- natural hard-negative training;
- direct relation to retrieval/ranking evaluation.

Questions to test:

- shared vs separate projection heads;
- tied vs untied backbone;
- dot product vs learned bilinear scorer;
- pooling strategy;
- evidence concatenation vs evidence fusion;
- action-description sensitivity;
- cache correctness under action revision changes.

## Architecture C — packed one-pass decision model

A decoder or encoder reads state plus multiple typed questions in one forward pass. Answer positions are restricted to the legal options.

This is both a serious candidate and a required baseline because current open System-One-style models demonstrate strong performance with answer-slot probability readout.

## Evidence module candidates

Evidence grounding must be evaluated with evidence perturbations.

Candidate mechanisms:

- evidence concatenated into state;
- independent evidence embedding with tri-linear scoring;
- evidence-attention pooling over multiple passages;
- top-k evidence aggregation;
- evidence-support auxiliary head.

The final paper may only claim evidence grounding if positive evidence, irrelevant evidence, contradictory evidence, and evidence-swap tests show the expected directional behavior.

## Abstention design

GAX separates:

- `p(action | state, evidence)`
- `p(sufficient | state, evidence)`
- decision policy `pi(p_action, p_sufficient, calibration_state)`

Candidate abstention signals:

- learned sufficiency head;
- entropy/margin baseline;
- conformal or calibration-set threshold;
- ensemble/MC uncertainty baseline where computationally feasible.

The learned head must beat confidence-only controls at matched coverage to justify inclusion.

## Calibration

Calibration is split into:

- model probability quality;
- post-hoc calibration;
- deployment decision threshold.

These must not be conflated.

Allowed calibration methods for comparison:

- temperature scaling;
- vector/Dirichlet-style calibration if data volume permits;
- isotonic regression only with sufficient calibration data;
- task-conditional calibration when preregistered and not test-fit.

Test labels may never select calibration parameters.

## Counterfactual consistency

Each counterfactual group includes:

- original state;
- clinically material edit;
- expected direction or changed action;
- one or more irrelevant edits.

Metrics:

- target flip/sensitivity rate;
- probability delta in expected direction;
- irrelevant-edit stability;
- calibration before/after perturbation.

## FHIR path

GAX should not learn accidental JSON serialization quirks.

FHIR experiments compare:

- raw canonical JSON;
- normalized clinical text;
- schema-aware flattened representation;
- selected-resource view.

The canonicalizer is versioned and independently tested.

## Serving

Target interfaces:

- in-process Python API;
- `POST /v1/systemone` compatibility layer for common typed-decision clients;
- `POST /v1/health/decide`;
- `POST /v1/health/verify`;
- `POST /v1/health/fhir`.

No serving optimization is allowed to change numerical outputs beyond a declared tolerance without a fidelity test.

## Model sizes

Initial research targets:

- **GAX-base:** compact model intended for accessible fine-tuning and serving.
- **GAX-large:** larger encoder candidate if it provides a justified Pareto improvement.
- optional decoder baseline/variant only if it materially improves the scientific result.

The project will not scale parameter count merely to win an accuracy table; efficiency is part of the research question.
