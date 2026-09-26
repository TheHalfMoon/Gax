# GAX v0 Model Card

Model: **GAX Bilinear v0**  
Architecture ID: `gax-bilinear-v0`  
Feature revision: `sha256-word-v0.1`  
Status: **P03 research reference / not clinically validated**

## Model description

GAX Bilinear v0 is a small non-generative typed-decision reference model used to qualify the GAX training, checkpoint, deterministic inference, and GAXBench integration path.

It maps a benchmark-visible state/evidence representation and caller-provided action descriptions to a normalized probability distribution over exactly those actions.

The model does not generate clinical text or arbitrary actions.

## Intended use

Permitted intended uses in P03:

- research software development;
- deterministic unit/integration testing;
- checkpoint and reproducibility testing;
- synthetic typed-decision experiments;
- validating GAXBench model integration.

## Out-of-scope use

Do not use GAX Bilinear v0 for:

- diagnosis;
- treatment selection;
- prescribing;
- triage;
- emergency guidance;
- patient-facing advice;
- autonomous clinical actions;
- clinical risk estimation;
- claims of medical accuracy, safety, calibration, or superiority.

## Training data

The P03 qualification path uses only the repository's hand-authored CC0 abstract `alpha`/`beta` fixture:

- `tests/fixtures/gax_v0_train.jsonl`
- `tests/fixtures/gax_v0_validation.jsonl`

The fixture contains no real patient data, PHI, medical threshold, clinical label, or external corpus.

Future GAX research models may use separately governed biomedical/clinical datasets only after license, provenance, leakage, and PHI review.

## Architecture

The reference model uses deterministic SHA-256 hashed word features for state/evidence and action descriptions, plus a trainable bilinear matrix:

```text
score(state, action) = s^T W a
```

Action scores are normalized with softmax.

Action IDs are output/schema identifiers and are not used as semantic action text.

## Uncertainty and abstention

GAX Bilinear v0 does not implement learned information sufficiency or a native abstention policy.

A high action probability must not be interpreted as permission to act. Learned sufficiency and matched-coverage selective prediction belong to later research phases.

## Evaluation

P03 evaluation is infrastructure qualification only. Synthetic smoke results must not be quoted as clinical performance.

Any future public performance claim must follow `docs/research/REPRODUCIBILITY.md` and be tied to frozen datasets, revisions, raw artifacts, failure counts, and statistical uncertainty.

## Reproducibility

The checkpoint binds:

- architecture and feature revision;
- model configuration;
- learned parameters;
- training seed;
- training-data manifest hash;
- completed epoch count;
- canonical payload integrity digest.

The GAX adapter also exposes model revision and checkpoint SHA-256 through the existing baseline identity/evidence path.

## Limitations

This reference model has deliberately limited representational capacity. It is not a clinical language model, does not establish the final GAX architecture, and is not evidence that hashed bilinear representations are sufficient for medical decision making.

Its role is to make the model lifecycle falsifiable and reproducible before higher-capacity clinical encoders and ECAL objectives are tested.

## License

GAX source code is Apache-2.0. The P03 abstract fixture is CC0-1.0. Future model weights and datasets must state their licenses independently.
