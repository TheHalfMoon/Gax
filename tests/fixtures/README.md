# Synthetic GAXBench fixtures

The JSONL files in this directory are hand-authored synthetic smoke fixtures.
They do not represent real patients, clinical guidance, validated medical thresholds,
or evidence for patient care.

The fixture data in this directory is dedicated to the public domain under CC0-1.0.
The surrounding GAX source code remains Apache-2.0.

The `items.jsonl` / `predictions.jsonl` fixture uses an abstract `synthetic_signal`
to exercise schema, probability, calibration, selective-risk, and abstention paths.
It has no medical unit, clinical threshold, or patient-care meaning.

The `gax_v0_train.jsonl` / `gax_v0_validation.jsonl` fixture uses only abstract
`alpha` and `beta` signals and action descriptions. It exists to prove that the P03
reference model can train, checkpoint, reload, infer, and pass through GAXBench.
Its results are prohibited from clinical, medical-performance, or paper benchmark claims.
