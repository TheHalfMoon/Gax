# P06 Evidence Interventions and Counterfactual Robustness

Status: **implementation grain SG-000009 / paper mechanisms defer-real-data**

P06 asks whether a decision model changes for the **right reasons**.

```text
uses evidence != merely receives evidence
material sensitivity != irrelevant-edit instability
counterfactual robustness != causal identification != clinical safety
```

The P06 evaluator is model-agnostic: any model that emits the canonical GAXBench `Prediction` contract can be evaluated under the same pair manifest and metric implementation.

## Why paired evaluation

Ordinary accuracy can hide brittle behavior. A model may answer the base item correctly but fail after one material fact changes. Conversely, a model may appear sensitive because it changes under formatting or irrelevant context.

P06 therefore evaluates explicit **base/intervention pairs** with a declared expected relation.

Supported v0.1 relations:

- `same` — action should remain unchanged;
- `flip` — gold action must change;
- `directional-only` — a declared action probability should move in a direction without requiring a top-1 flip;
- `abstain` — the intervention should trigger explicit abstention.

## Lineage contract

Every pair must:

1. reference two distinct benchmark item IDs;
2. preserve `source_id`;
3. stay in exactly one split;
4. share a non-null `counterfactual_group`;
5. use the same action IDs in P06 v0.1;
6. use gold actions consistent with `same` or `flip` when those relations are declared.

The identical-action-set constraint is deliberate. Candidate removal/addition changes the probability simplex and makes direct TV/JSD/action-flip comparisons ambiguous. That research question is deferred to a separately governed extension.

## Distribution metrics

### Total variation

For matching action distributions `p` and `q`:

```text
TV(p, q) = 0.5 * sum_a |p(a) - q(a)|
```

Range: `[0, 1]`.

### Jensen-Shannon divergence

P06 uses natural logarithms:

```text
m = 0.5 * (p + q)
JSD(p, q) = 0.5 KL(p || m) + 0.5 KL(q || m)
```

Range for normalized distributions: `[0, ln(2)]`.

The log base is part of the versioned metric contract.

## Counterfactual metrics

P06 reports separately:

- base accuracy;
- intervention accuracy;
- paired robust accuracy (`base correct AND intervention correct`);
- number of flip pairs;
- number of base-correct flip pairs;
- required-flip success;
- base-conditioned counterfactual failure / Bias Trap Rate-style metric.

The conditional failure denominator contains only flip pairs for which the model was correct on the base/control item. This prevents an already-wrong base answer from being counted as a new counterfactual failure.

## Stability metrics

For `same` relations P06 reports:

- top-1 agreement;
- TV/JSD shift;
- maximum action-probability delta;
- stability violation rate against a declared TV threshold.

The threshold is an input to the evaluator and must be fixed before frozen evaluation. P06 does not optimize this threshold on final test.

`materiality = irrelevant` is only valid with `expected_relation = same`.

## Directional metrics

A pair can declare:

- `target_action`;
- `target_direction = increase | decrease | same`;
- `expected_sufficiency_direction`;
- `expected_evidence_support_direction`.

A declared score direction fails closed if the required base/intervention score is absent. The evaluator never silently removes that pair from the denominator.

## Determinability under missing information

Evidence or context removal does **not** automatically imply abstention.

P06 includes two distinct concepts:

1. **underdetermined after removal** — sufficiency should fall and abstention may be correct;
2. **still determinable after removal** — the correct action remains available and the model should not over-refuse.

This pressure test is motivated by recent incomplete-information benchmarks such as ClinDet-Bench. Related work guides falsification design; it is not evidence for GAX performance.

## Evidence sensitivity is not automatically safety

P06 does not reward blind faithfulness to any supplied evidence.

Counterfactual or implausible evidence can make a model more faithfully wrong. The protocol therefore permits interventions where the correct behavior is lower evidence support, lower sufficiency, or explicit abstention. Source validity and plausibility remain part of the governed data contract for later real-evidence experiments.

This boundary is pressure-tested by recent work including MedCounterFact and heterogeneous-evidence evaluations such as MEDSYN.

## Final-test protection

`gaxbench interventions-evaluate` rejects any input item with `split = test` during P06.

The final test split remains sealed until P08, when model revisions, intervention manifest, stability thresholds, benchmark version, baselines, and statistical protocol are frozen.

## CLI

```bash
gaxbench interventions-evaluate \
  --items tests/fixtures/p06_items.jsonl \
  --predictions tests/fixtures/p06_predictions.jsonl \
  --manifest tests/fixtures/p06_interventions.json \
  --stability-tv-threshold 0.05
```

Output contains:

- manifest SHA-256;
- declared stability threshold;
- aggregate metrics;
- per-family metrics;
- every pair-level record.

Per-family reporting is mandatory so one strong intervention family cannot hide a catastrophic regression in another.

## Synthetic fixtures

The P06 fixtures are hand-authored abstract mechanics tests only:

- material route flip;
- irrelevant formatting edit;
- determinable missing-information case;
- abstract contradictory-evidence abstention case;
- abstract supporting-evidence directional case.

They contain no patient data, PHI, real medical condition, medication, treatment, threshold, or clinical guidance.

Synthetic success proves only that the evaluator and invariants behave as specified.

## Paper gate

All P06 paper mechanisms start as:

```text
defer-real-data
```

A mechanism can become `keep` only on licensed, reviewed, leakage-audited development evidence when:

1. material sensitivity/evidence dependence improves under paired evaluation;
2. irrelevant-edit stability does not regress beyond the preregistered tolerance;
3. the effect survives confidence intervals and matched baselines;
4. hidden-label/test leakage audits are clean;
5. evidence sensitivity is not achieved by blindly accepting implausible evidence;
6. requested/completed/failed denominators remain explicit.

Null results remain valid research results and stay in the ledger.

## Related pressure tests

The P06 literature map must be refreshed again before paper freeze. Current pressure-test families include:

- Med-PRM — guideline/evidence-verified process rewards;
- MediEval — patient-contextual factual/counterfactual evaluation;
- MedEinst and MamaBench — paired counterfactual robustness and Bias Trap Rate;
- ASCENT — stepwise diagnostic reasoning under incomplete information;
- ClinDet-Bench — determinability under missing information;
- MedCounterFact — faithfulness versus safety under counterfactual evidence;
- MEDSYN — heterogeneous evidence sensitivity.

No external reported result is treated as a GAX result.

## Exit gate

P06 closes only after:

1. strict manifest/schema, lineage, metrics, CLI, registry, fixtures, tests, and ledger are complete;
2. exact-head CI passes Ruff, mypy strict, pytest, and compileall on Linux/Windows × Python 3.11/3.12;
3. implementation merges with an expected-head guard;
4. post-main CI succeeds;
5. a separate closeout marks SG-000009 `PROVEN` while retaining synthetic-only paper decisions as `defer-real-data`;
6. the canonical frontier advances to P07 FHIR.
