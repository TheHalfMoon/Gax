# P08 Statistical Uncertainty and Evidence Aggregation

Status: **SG-000012 implementation grain / final test sealed**

This document freezes the statistical and evidence-aggregation mechanics that GAX will use before any real final-test opening.

SG-000012 is research infrastructure. It does not contain a clinical benchmark result and does not authorize a model, safety, superiority, FHIR, evidence-grounding, abstention, or SOTA claim.

## Why this is frozen before final testing

A paper can be biased even when model weights are frozen if analysis choices remain movable after test results are visible. GAX therefore fixes the following mechanics first:

- paired uncertainty estimation;
- evidence-ranking definitions and tie behavior;
- calibration/reliability bin semantics;
- failure denominators;
- table/figure/report provenance;
- the primary-comparison registry and deterministic report representation.

Changing one of these definitions after final-test access requires a new experiment revision and cannot silently rewrite the original result.

## Paired bootstrap confidence intervals

`paired_bootstrap_mean_difference` compares two systems on the same ordered items.

For paired values `a_i` and `b_i`, the reported point estimate is:

```text
mean_i(a_i - b_i)
```

The interval is a deterministic **percentile bootstrap** over paired item differences:

1. compute the per-item differences once;
2. sample `N` difference indices with replacement for each replicate;
3. compute the replicate mean;
4. repeat using the caller-frozen pseudo-random seed;
5. take the requested two-sided percentile interval using linear quantile interpolation.

The implementation requires at least 1,000 replicates. Paper evaluation must freeze the actual replicate count, seed, and CI level in the P08 manifest before final-test access.

This is not a BCa interval and must not be described as one.

## Evidence AUROC

Evidence AUROC uses pairwise concordance:

- positive score greater than negative score: `1.0` credit;
- equal positive/negative score: `0.5` credit;
- positive score lower than negative score: `0.0` credit.

The metric is undefined when there are no positive labels or no negative labels. Undefined cases are represented explicitly with `value = null` plus a reason; they are never converted to zero, one, or a silently excluded row.

## Evidence AUPRC

AUPRC uses deterministic score-grouped threshold steps.

All examples with the same score enter the threshold set together. At each distinct descending score:

```text
recall = TP / positives
precision = TP / (TP + FP)
area += (recall - previous_recall) * precision
```

This is a step-area convention over grouped thresholds. It is intentionally versioned because alternative interpolation conventions can produce different values.

AUPRC is undefined when no positive labels exist. When all examples are positive, the grouped-threshold convention yields `1.0`.

## Reliability bins

Reliability artifacts use equal-width bins over `[0, 1]` with the same assignment convention as GAXBench fixed-bin ECE:

```text
index = min(int(confidence * bins), bins - 1)
```

Each artifact records:

- bin index;
- lower edge;
- upper edge;
- whether the upper edge is inclusive;
- count;
- mean confidence;
- empirical accuracy;
- absolute calibration gap.

Empty bins are retained with null summary statistics. Dropping empty bins would make figures harder to reproduce and can hide differences in plotting code.

## Failure-preserving aggregation

A requested evaluation item has exactly one run outcome.

Supported statuses are:

- `success`;
- `timeout`;
- `oom`;
- `parse_failure`;
- `inference_failure`;
- `evaluation_failure`;
- `blocked`.

A successful outcome requires a finite value and no failure detail. A non-success outcome requires detail and must not carry a metric value.

Aggregation records:

```text
requested
completed
failed
status_counts
failure_item_ids
```

If any requested item failed, the generic aggregate mean is withheld rather than computed on the successful subset. Task-specific paper analyses may define a different failure treatment only if that policy was frozen in the P08 protocol before final-test access.

This implements the project rule:

> missing/failed inference != silent exclusion

## Derived artifact manifests

Every paper-facing table, figure, or deterministic report should receive a derived-artifact manifest.

The manifest binds:

- artifact ID and kind;
- generator revision;
- sorted source run IDs;
- sorted evidence-packet IDs;
- SHA-256 for every declared source file;
- output SHA-256;
- canonical manifest digest.

The digest is computed over canonical JSON with the digest field set to null. A changed source, output, identifier, or generator revision therefore changes the evidence identity.

## Primary comparison registry

Primary comparisons are declared before final evaluation.

Each registry entry names:

- comparison ID;
- metric;
- system A;
- system B;
- metric direction;
- paired status, fixed to `true` for this contract;
- multiplicity family.

Comparison IDs must be unique and sorted. The real P08 freeze will populate the actual primary comparisons and multiplicity families; SG-000012 only freezes the data model and report mechanics.

## Comparison result set

A result set is tied to one `experiment_revision` and contains one result per comparison ID.

A `complete` result requires:

- `requested == completed`;
- `failed == 0`;
- nonzero completed count;
- finite system estimates;
- a finite difference;
- a finite confidence interval and valid CI level;
- no failure reason.

A `blocked` or `undefined` result must carry an explicit reason and must not carry fabricated numeric estimates.

The primary report is generated only when:

- registry and result-set experiment revisions match;
- result IDs exactly cover registry IDs with no missing or extra comparison;
- the report binds a 40-character repository commit SHA.

The serialized report is stable sorted JSON with a canonical report digest.

## CLI

The existing `gax-p08` command now includes pre-test statistical utilities.

### Paired bootstrap

Input:

```json
{
  "values_a": [1.0, 0.0, 1.0],
  "values_b": [0.0, 0.0, 1.0]
}
```

Command:

```bash
gax-p08 stats-bootstrap \
  --input paired.json \
  --replicates 10000 \
  --seed 1729 \
  --ci-level 0.95
```

### Evidence ranking

```json
{
  "scores": [0.9, 0.8, 0.8, 0.1],
  "labels": [true, true, false, false]
}
```

```bash
gax-p08 evidence-rank --input evidence.json
```

### Reliability bins

```json
{
  "confidences": [0.1, 0.2, 0.9],
  "correctness": [true, false, true]
}
```

```bash
gax-p08 reliability --input calibration.json --bins 15
```

### Primary comparison report

```bash
gax-p08 comparison-report \
  --registry primary-comparisons.json \
  --results primary-results.json \
  --repo-revision <40-char-git-sha> \
  --output primary-report.json
```

The command will not overwrite the registry or result-set input file.

## Final-test boundary

SG-000012 does **not** call `authorize_final_test`, does not change the sealed P08 fixture, and does not populate any real final-test result.

The next P08 stage must first populate and audit the real development/freeze inventory:

- immutable benchmark/split manifests;
- licenses and redistribution status;
- leakage evidence;
- GAX checkpoint revisions and training seeds;
- baseline model/tokenizer/source revisions and qualification status;
- calibration split and policy;
- ECAL development keep/reject decisions;
- FHIR representation candidates;
- hardware and timing protocol.

Only after those inputs are complete, audited, clean-tree bound, and digest-authorized may final-test access change from `sealed` to `authorized`.
