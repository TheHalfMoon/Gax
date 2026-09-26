# P03 GAX v0 Reference Model

Status: **implementation grain SG-000006**

P03 establishes the first genuine trainable, non-generative GAX model. It is a reference architecture for lifecycle and reproducibility qualification, not the final architecture claimed by the research paper.

## Purpose

P00-P02 froze the research, benchmark, and comparison contracts before model work. P03 now proves that GAX can:

1. train a typed decision model;
2. save an integrity-bound checkpoint;
3. reload the checkpoint without changing predictions;
4. emit normalized probabilities over caller-provided actions;
5. run through the existing GAXBench evaluator and failure accounting;
6. reproduce the complete path on ordinary CPU CI without paid compute.

P03 does **not** test clinical capability.

## Architecture

Architecture ID: `gax-bilinear-v0`

The model is a small deterministic bilinear state/action scorer:

```text
state/evidence --SHA-256 hashed features--> s
                                          |
                                          |  s^T W a
                                          v
action description --hashed features----> a ---> action logit ---> softmax
```

For action `a_i`:

```text
logit_i = s^T W a_i
p(a_i | state, evidence) = softmax(logits)_i
```

`W` is trainable. The feature map is fixed and versioned as `sha256-word-v0.1`.

The action ID is not used as semantic training text. It is preserved only as the stable schema key for the output probability. This prevents the reference model from treating arbitrary action labels as clinical meaning.

## Why this model is intentionally small

P03 is a mechanics gate. Pulling a large transformer into the core package here would conflate three questions:

- whether GAX's model lifecycle is reproducible;
- whether a particular clinical backbone is useful;
- whether ECAL improves the final research architecture.

The first question is answered in P03 with a dependency-light reference model. Backbone selection and ECAL mechanisms remain controlled research questions for later phases.

The P03 reference must never be described as the final GAX-base paper model unless later evidence independently justifies that conclusion.

## Input boundary

The model sees only benchmark-visible fields:

- `state`;
- evidence text/structured values and evidence relation;
- action descriptions.

It does not consume:

- gold action labels at inference;
- source/provenance identity as a predictive feature;
- benchmark split name as a predictive feature;
- hidden rationale;
- patient identifiers;
- action ID text as semantic content.

Actions are sorted by ID internally before scoring so a caller's action-list permutation cannot change ID/probability alignment through ordering alone.

## Training

`train_gax_v0` accepts only `train` items as training examples. Optional validation inputs must have split `validation`.

The trainer rejects:

- empty datasets;
- duplicate item IDs;
- non-training split values in the training set;
- missing gold actions;
- non-finite losses or weights.

Training uses deterministic fixed-seed initialization and epoch shuffling. The objective in P03 is ordinary multiclass negative log likelihood. ECAL is intentionally absent until P04.

The synthetic qualification fixture contains abstract `alpha` and `beta` signals only. It has no clinical concept, medical threshold, patient, or PHI.

## Checkpoint contract

Schema version: `0.1`

The checkpoint is canonical JSON containing:

- architecture ID;
- feature revision;
- full model config;
- learned weight matrix;
- training seed;
- training-data manifest SHA-256;
- completed epoch count.

The payload is wrapped by a SHA-256 over canonical JSON. Loading fails closed if:

- the envelope shape is wrong;
- the payload digest changes;
- schema version is unsupported;
- architecture or feature revision is unsupported;
- model dimensions disagree with config;
- any weight is non-finite;
- training seed disagrees with model config.

The file itself also receives a SHA-256 for evidence packets and adapter identity.

## Inference and abstention boundary

P03 emits action probabilities only.

It deliberately leaves:

```text
information_sufficiency = null
abstain = false
```

rather than pretending that top-1 confidence is epistemic sufficiency. Native learned sufficiency and selective policy are P05 research tasks and must beat confidence-only controls at matched coverage before they become a claimed GAX capability.

## GAXBench integration

`GaxV0Adapter` implements the same adapter contract as all P02 baselines. Evaluation therefore passes through:

```text
run_baseline -> prediction validation -> P01 metric code -> evidence/failure accounting
```

There is no alternate GAX-only metric path.

## CLI

Train the abstract smoke model:

```bash
gax train \
  --train-items tests/fixtures/gax_v0_train.jsonl \
  --validation-items tests/fixtures/gax_v0_validation.jsonl \
  --checkpoint /tmp/gax-v0.json \
  --feature-dim 16 \
  --learning-rate 0.4 \
  --epochs 60 \
  --seed 7
```

Evaluate it through GAXBench:

```bash
gax evaluate \
  --items tests/fixtures/gax_v0_validation.jsonl \
  --checkpoint /tmp/gax-v0.json
```

Numbers produced by these fixtures are smoke-test artifacts only and are prohibited from clinical or paper performance claims.

## Qualification requirements

P03 implementation qualification requires:

- training loss decreases on the abstract learnable fixture;
- the fixture mapping is learned end to end;
- fixed-seed training is deterministic within the declared protocol;
- checkpoint round trip preserves predictions;
- payload tampering is detected;
- action permutation preserves action-ID/probability alignment;
- a non-training split is rejected by the trainer;
- the loaded model runs through the existing GAXBench runner;
- train/evaluate CLI passes end to end;
- Ruff, mypy strict, pytest, and compileall pass on Python 3.11/3.12 on Linux and Windows.

## Scientific limitations

The reference model is not evidence of:

- clinical correctness;
- clinical calibration;
- safe deployment;
- superior abstention;
- evidence grounding;
- FHIR competence;
- superiority to Jev, CLM, Laya, decider, clinical encoders, or LLM baselines;
- SOTA performance.

Those claims require later frozen experiments.
