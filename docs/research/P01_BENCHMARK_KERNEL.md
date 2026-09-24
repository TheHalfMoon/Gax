# P01 Benchmark Kernel

Status: **implementation grain SG-000002**

P01 turns the GAXBench specification into a small, strict, reproducible evaluation kernel before any model training begins.

## Scope

The kernel provides:

- versioned benchmark and prediction schemas;
- deterministic JSONL I/O;
- exact item/prediction ID alignment;
- action-key validation;
- normalized probability validation;
- action accuracy;
- negative log likelihood;
- multiclass Brier score;
- fixed-bin expected calibration error;
- risk-coverage curves;
- discrete area under the risk-coverage curve;
- risk at fixed target coverage;
- actual abstention coverage and committed risk;
- abstention precision/recall/F1;
- unsafe-commit and over-abstain rates;
- sufficiency Brier/ECE when sufficiency scores are available;
- file and canonical-JSON SHA-256 provenance;
- a CLI evaluator;
- deterministic synthetic smoke fixtures.

No real clinical dataset enters P01.

## Metric definitions

### Accuracy

For each labeled item, select the action with maximum predicted probability. Ties are broken deterministically by action ID through Python tuple ordering.

```text
accuracy = correct top-1 actions / labeled items
```

Abstention does not erase the underlying action prediction. Action accuracy is reported independently from the deployment abstention policy.

### Negative log likelihood

For item `i` with gold action `y_i`:

```text
NLL = -(1 / N) * sum_i log(max(p_i[y_i], 1e-12))
```

The epsilon prevents numerical infinity in the reporting function. A model assigning exactly zero probability to the gold action is still heavily penalized.

### Multiclass Brier score

For action set `A_i`:

```text
Brier_i = sum_{a in A_i} (p_i[a] - 1[a = y_i])^2
Brier = mean_i Brier_i
```

This is the summed multiclass convention, not divided by the number of classes. The convention must remain stable within a benchmark version.

### Expected calibration error

The kernel uses equal-width confidence bins on `[0, 1]`:

```text
ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|
```

Default bins: 15.

ECE is bin-sensitive and must never be the only calibration metric. NLL and Brier are always reported beside it in paper-quality evaluation.

### Selection score

Risk-coverage evaluation needs one scalar per item.

- If **every** prediction supplies `information_sufficiency`, that score is used.
- Otherwise the entire run falls back to maximum action probability.

The evaluator never mixes sufficiency and max-probability scores within one curve.

### Risk-coverage curve

Sort items by selection score descending. Score ties preserve original benchmark order.

For every prefix of size `k`:

```text
coverage_k = k / N
risk_k = errors in first k / k
```

### Discrete AURC

GAXBench v0.1 defines:

```text
AURC = mean_k risk_k
```

over all attainable prefix coverages. This discrete definition is versioned so future implementations cannot silently switch integration conventions.

### Risk at target coverage

For target `c`, select the first attainable prefix whose coverage is at least `c` and report its risk.

The kernel reports risk@50%, risk@80%, and risk@90%.

### Actual abstention policy

The `abstain` field represents the model/deployment policy.

```text
actual_coverage = non-abstained items / N
committed_risk = 1 - accuracy(non-abstained items)
```

If every item is abstained, committed accuracy/risk are `null`, not zero.

### Abstention quality

Positive class means **should abstain**, defined by `gold.sufficient == false`.

Reported:

- precision;
- recall;
- F1;
- unsafe-commit rate = committed / insufficient items;
- over-abstain rate = abstained / sufficient items.

When every record also has `information_sufficiency`, the kernel reports Brier and ECE for sufficiency itself.

## Failure policy

The evaluator fails closed instead of silently changing the denominator.

It raises on:

- duplicate item IDs;
- duplicate prediction IDs;
- missing predictions;
- extra predictions;
- unlabeled items in action evaluation;
- prediction action keys that differ from the item's exact action set;
- probabilities that are non-finite, outside `[0, 1]`, or fail to sum to one within `1e-6`;
- non-finite selection scores.

This is intentional research behavior.

## Schemas

Published snapshots:

- `schemas/gaxbench-item-v0.1.json`
- `schemas/gaxbench-prediction-v0.1.json`

CI tests the snapshots against the Pydantic models to prevent accidental interface drift.

## CLI

```bash
gaxbench evaluate \
  --items tests/fixtures/items.jsonl \
  --predictions tests/fixtures/predictions.jsonl \
  --ece-bins 10
```

The CLI writes deterministic JSON metrics to stdout.

Exact cross-split leakage auditing is also exposed directly:

```bash
gaxbench audit --items tests/fixtures/items.jsonl
```

The audit reports duplicate item IDs and exact cross-split collisions in source IDs, canonical input fingerprints, and counterfactual groups. It intentionally does not claim semantic near-duplicate detection.

## Qualification

Historical local smoke checks were used while authoring, but they are not merge evidence because the branch continued to evolve afterward.

**Exact-head GitHub CI is authoritative** and must run:

- Ruff;
- mypy strict;
- pytest;
- compileall;

on Python 3.11 and 3.12 for Linux and Windows.

## Deliberate omissions

P01 does not yet implement:

- AUROC/AUPRC for evidence support;
- bootstrap confidence intervals;
- paired significance tests;
- benchmark dataset adapters;
- model adapters;
- real clinical data;
- leaderboard aggregation.

Those belong to later governed grains. P01's job is to make the denominator, probabilities, selective-risk definitions, and provenance trustworthy first.
