# P05 Native Information Sufficiency and Selective Abstention

Status: **CLOSED_CANONICAL / SG-000008 PROVEN / learned-sufficiency paper benefit remains defer-real-data**

P05 separates **action preference** from **permission to act**.

```text
action_distribution != information_sufficiency != permission_to_act
```

A high probability on one allowed action can coexist with insufficient information. P05 therefore evaluates a separately learned information-sufficiency signal against strong confidence-only selector controls at matched coverage.

Synthetic fixtures in this phase are implementation evidence only. They cannot support a clinical-risk, safety, SOTA, or paper-superiority claim.

## Selector controls

P05 freezes four selector definitions.

### Max probability

```text
score = max_a P(a | state)
```

This is the simplest confidence-only control.

### Normalized entropy confidence

For `K > 1` actions:

```text
H(p) = -sum_a p(a) log p(a)
score = 1 - H(p) / log(K)
```

For one legal action the score is `1.0`.

### Top-1 / top-2 margin

```text
score = p_(1) - p_(2)
```

where `p_(1) >= p_(2)`. For one legal action the score is `1.0`.

### Learned information sufficiency

The first learned mechanism is intentionally low complexity: a deterministic logistic model over:

- the same model-visible hashed state representation used by the P03 reference model;
- maximum action probability;
- normalized entropy confidence;
- top-1/top-2 margin.

The action model is frozen while the sufficiency model trains.

The learned model is **not allowed** to observe:

- `gold.sufficient` at inference;
- gold action;
- source/split identity;
- task-family bookkeeping;
- provenance construction metadata;
- benchmark-only evidence relation labels;
- final-test annotations.

`render_model_state` remains the visibility boundary, so evidence text/structured content may be visible while `evidence.relation` remains hidden supervision metadata.

## Training contract

`train_information_sufficiency` accepts only `train` examples with explicit sufficiency labels. The frozen action model produces its ordinary action distribution; P05 never updates action-model weights during sufficiency training.

The reference learned selector uses binary negative log likelihood with optional L2 regularization and deterministic seeded example ordering.

This is a mechanism baseline, not a claim that logistic sufficiency is the final GAX architecture.

## Calibration-set coverage policy

Threshold selection is intentionally separated from validation evaluation.

For selector `S` and target coverage `c`:

1. compute selector scores on the dedicated `calibration` split;
2. sort descending;
3. select the score at `ceil(c * N)` as the inclusive commit threshold;
4. freeze the threshold;
5. evaluate on the separate `validation` split.

The calibration threshold uses scores only. It does not need calibration gold sufficiency or action labels, and it never uses final-test labels.

Score ties can make realized coverage exceed the target. This is reported rather than silently broken with labels.

P05 does **not** claim a formal conformal guarantee. A later method may do so only if its exchangeability/risk-control assumptions and theorem conditions are implemented and verified.

## Matched-coverage evaluation

Every selector is evaluated on identical validation examples.

Required outputs include:

- full action accuracy independent of abstention;
- requested/completed/failed counts;
- actual policy coverage;
- committed risk;
- risk-coverage curve;
- AURC;
- risk at target coverage;
- risk@50/80/90;
- abstention precision/recall/F1;
- unsafe-commit rate;
- over-abstention rate;
- learned-sufficiency probability calibration metrics where available.

Canonical `risk_coverage_curve`, `risk_at_coverage`, `evaluate_abstention`, and `expected_calibration_error` implementations are reused rather than forked into P05-specific metric definitions.

A selector cannot win merely by abstaining more often. The paper-facing comparison is **risk at matched coverage**, with over-abstention shown beside unsafe commits.

## Final-test protection

P05 mechanism qualification accepts `validation` items and rejects `test` items in the selector-evaluation path.

The final test split is reserved for the P08 frozen paper evaluation after model, selector, threshold/calibration procedure, benchmark version, and baseline revisions are frozen.

## Synthetic fixtures

The P05 fixtures are hand-authored CC0 abstract states with fields such as:

- `scope = in | out`;
- `completeness = complete | missing-critical`;
- `consistency = coherent | contradictory`;
- an abstract `left | right` route.

They contain no real patient data, PHI, medical thresholds, diagnosis rules, treatment guidance, or clinical source text.

Files:

```text
tests/fixtures/p05_train.jsonl
tests/fixtures/p05_calibration.jsonl
tests/fixtures/p05_validation.jsonl
```

## Reproducibility surface

P05 exposes two CLI commands.

```bash
gax selective-manifest \
  --train-items tests/fixtures/p05_train.jsonl \
  --calibration-items tests/fixtures/p05_calibration.jsonl \
  --validation-items tests/fixtures/p05_validation.jsonl \
  --checkpoint /path/to/gax-v0.json \
  --git-sha <exact-40-char-sha> \
  --compute-provenance <description> \
  --target-coverage 0.80
```

and:

```bash
gax selective-evaluate \
  --train-items tests/fixtures/p05_train.jsonl \
  --calibration-items tests/fixtures/p05_calibration.jsonl \
  --validation-items tests/fixtures/p05_validation.jsonl \
  --checkpoint /path/to/gax-v0.json \
  --git-sha <exact-40-char-sha> \
  --compute-provenance <description> \
  --target-coverage 0.80
```

The manifest binds:

- exact git SHA;
- compute provenance;
- action-model revision and feature revision;
- train/calibration/validation hashes;
- sufficiency configuration and hash;
- selector set;
- target coverage;
- threshold protocol.

The CLI additionally emits the checkpoint artifact SHA-256.

## Paper gate

The learned-sufficiency paper decision remains:

```text
defer-real-data
```

On licensed, leakage-audited development evidence, learned sufficiency may become `keep` only if it improves the preregistered matched-coverage selective-risk metric over the strongest confidence-only control without unacceptable over-abstention or action-quality regression.

Null or negative results remain publishable research evidence and must stay in the ledger.

## Canonical qualification

P05 implementation evidence is bound to:

- implementation PR: `#22`;
- exact implementation head: `309e7c108f4dab87f80d2ab24849d9c3032102db`;
- exact-head CI: run `36251925614` — SUCCESS on Linux/Windows × Python 3.11/3.12;
- implementation merge: `ced14e8e8f599f8022a273c762fda56452c335c6`;
- post-main CI: run `36252259764` — SUCCESS on Linux/Windows × Python 3.11/3.12.

The Windows/Python 3.11 exact-head job recorded Ruff success, mypy strict success with no issues in 17 source files, **84 passing tests**, and compileall success. The other three exact-head matrix jobs also completed successfully, followed by a fully green post-main matrix.

P05 is therefore closed as an **implementation and selective-evaluation framework result**. Learned information sufficiency is not promoted to a positive paper result from synthetic fixtures. The next frontier is P06 evidence interventions and counterfactual clinical robustness.
