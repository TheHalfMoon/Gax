# P04 — Evidence-Calibrated Action Learning (ECAL)

Status: **implementation grain SG-000007 / paper contribution not yet proven**

P04 turns ECAL from a working research idea into a falsifiable set of controlled mechanisms. It does not assume that any ECAL component improves GAX, and it does not treat synthetic mechanism tests as clinical or paper-performance evidence.

## Research boundary

P03 already trains a state-to-action model with multiclass negative log likelihood. That ordinary action NLL remains the reference objective and is **not** renamed as a new contrastive contribution.

P04 studies additional mechanisms:

```text
L = w_action * L_action
  + w_bidir * L_bidirectional
  + w_hard * L_hard_negative
  + w_evidence * L_evidence
  + w_proper * L_proper
```

Replay is a sampling policy, not a scalar loss.

Every auxiliary scalar term is independently zeroable. `w_action` remains strictly positive in P04.

## 1. Bidirectional multi-positive alignment

The added contrastive structure contains two directions:

1. current state -> compatible action representations;
2. current action representation -> compatible state representations.

The semantic positive key is the normalized **action description**, not the benchmark item ID or action ID. If multiple examples share the same semantic action description, they form a multi-positive set.

For logits `z` and positive index set `P`:

```text
L_multi = -log( sum_{j in P} exp(z_j) / sum_k exp(z_k) )
```

The gradient treats all positives as positives. Duplicate semantic actions are therefore not converted into false negatives.

## 2. Hard-negative ranking

For one labeled item, the hard-negative policy chooses the highest-scoring incorrect candidate from the caller-provided visible candidate set. Score ties use deterministic action ordering.

The margin objective is:

```text
L_hard = max(0, margin - score(gold) + score(negative))
```

The matched control uses the same objective weight and optimizer-step budget but chooses a seeded random incorrect candidate. This prevents a hard-negative claim from being compared against a weaker no-objective control.

## 3. Evidence intervention objective

`evidence.relation` is **supervision metadata only**. Before an evidence item enters the state renderer, its relation is replaced by `unknown`; the visible text or structured payload remains unchanged.

The objective compares the gold-action score under:

- evidence removed;
- one visible evidence item added.

Supervision behavior:

- `support`: encourage the gold-action score to increase by a declared margin;
- `contradict`: encourage the gold-action score to decrease by a declared margin;
- `irrelevant`: penalize score movement toward either direction;
- `unknown`: no evidence-supervision gradient.

Inference must remain invariant when only `evidence.relation` changes while visible content stays identical. This is a regression gate, not a paper-level evidence-faithfulness result.

## 4. Proper-scoring objective

The initial proper-scoring mechanism is the summed multiclass Brier score:

```text
L_Brier = sum_j (p_j - y_j)^2
```

For softmax probability `p_j`, error `e_j = p_j - y_j`, and

```text
m = sum_k p_k * e_k
```

the logit derivative used by P04 is:

```text
dL_Brier / dz_j = 2 * p_j * (e_j - m)
```

The test suite checks this analytic gradient against central finite differences. A later paper claim must report NLL, Brier, and ECE together; ECE alone is not sufficient evidence of calibration improvement.

## 5. Replay and retention

Replay uses **equal-step replacement**.

For a target schedule of `N` optimizer steps and replay ratio `r`, a deterministic subset of target positions is replaced by replay examples. The treatment does not silently receive additional optimizer steps.

P04 records:

- total optimizer steps;
- target steps;
- replay steps;
- target-data manifest;
- replay-data manifest;
- a frozen retention slice evaluated with the same canonical GAXBench runner used for the current development slice.

A replay mechanism cannot be called beneficial merely because retention improves if current-task quality collapses.

## Matched ablation contract

For every component `C`:

```text
CONTROL   = same architecture + same eligible data + same seed + same step budget - C
TREATMENT = CONTROL + C
```

The hard-negative experiment is stricter: the control uses a random-negative margin objective at the same weight instead of deleting the ranking objective.

The repository produces a deterministic P04 manifest that binds:

- exact git commit SHA;
- declared compute provenance;
- train data hash;
- development data hash;
- replay data hash when used;
- retention data hash when used;
- base model/training configuration;
- control and treatment ECAL configuration hashes;
- the declared development gate;
- the prohibition on final-test tuning.

`--git-sha` and `--compute-provenance` are required CLI inputs. They are not inferred silently. A paper-eligible run must use the exact code revision being executed and a truthful compute description.

## Paper decision policy

The P04 synthetic fixtures are sufficient to prove implementation properties such as:

- deterministic gradients and schedules;
- multi-positive semantics;
- hidden-label isolation;
- matched step accounting;
- canonical evaluator integration.

They are **not** sufficient to prove that an ECAL mechanism improves clinical decision quality.

Therefore the initial paper decision for every mechanism is:

```text
defer-real-data
```

A mechanism may become `keep` or `reject` only after the preregistered development gate is evaluated on a licensed, leakage-audited development source. Final benchmark test labels remain unavailable for that choice. Null or negative results stay in the decision ledger.

## Synthetic qualification fixtures

P04 ships four CC0 abstract fixture families:

- `ecal_train.jsonl` — repeated alpha/beta semantic actions plus evidence metadata;
- `ecal_validation.jsonl` — current-task development slice;
- `ecal_replay.jsonl` — prior gamma capability used only by replay;
- `ecal_retention.jsonl` — frozen prior-capability validation slice.

The tokens `alpha`, `beta`, `gamma`, `support`, and similar strings have no clinical meaning. The fixtures contain no patient data, medical thresholds, diagnoses, treatment recommendations, or real evidence sources.

## Reproducibility commands

The `gax` CLI exposes P04 manifest and matched-ablation commands. Every command requires explicit experiment context. A typical synthetic qualification run is:

```bash
gax ecal-manifest \
  --train-items tests/fixtures/ecal_train.jsonl \
  --validation-items tests/fixtures/ecal_validation.jsonl \
  --replay-items tests/fixtures/ecal_replay.jsonl \
  --retention-items tests/fixtures/ecal_retention.jsonl \
  --feature-dim 8 --epochs 3 --learning-rate 0.05 --seed 13 \
  --git-sha <EXACT_40_CHAR_COMMIT_SHA> \
  --compute-provenance "<hardware/runtime provenance>"
```

and:

```bash
gax ecal-ablate replay \
  --train-items tests/fixtures/ecal_train.jsonl \
  --validation-items tests/fixtures/ecal_validation.jsonl \
  --replay-items tests/fixtures/ecal_replay.jsonl \
  --retention-items tests/fixtures/ecal_retention.jsonl \
  --feature-dim 8 --epochs 3 --learning-rate 0.05 --seed 13 \
  --git-sha <EXACT_40_CHAR_COMMIT_SHA> \
  --compute-provenance "<hardware/runtime provenance>"
```

Numeric output from these fixtures is infrastructure qualification only and must not enter the paper as a clinical performance result.

## Exit gate

P04 is not closed by implementation alone. SG-000007 can become `PROVEN` only after:

1. every mechanism and matched-control path is unit tested;
2. hidden evidence labels are proven non-visible at inference;
3. the Brier gradient finite-difference test passes;
4. deterministic manifest and replay/retention accounting tests pass;
5. the decision ledger remains explicit about deferred or negative mechanisms;
6. exact-head Ruff, mypy strict, pytest, and compileall pass on Linux/Windows and Python 3.11/3.12;
7. the implementation merges with an exact-head guard;
8. post-main CI succeeds;
9. a separate canonical closeout advances the frontier to P05 native abstention.
