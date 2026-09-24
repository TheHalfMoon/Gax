# GAX

**Grounded Action eXpert for Health**

> Clinical decisions, not paragraphs. Know when not to act.

GAX is an open research project for **evidence-grounded, calibrated, non-generative clinical decision models**. The goal is to build a fast System-One-style decision layer for health AI that selects among explicit actions, reports calibrated uncertainty, verifies evidence support, and can abstain when the available information is insufficient.

GAX is **research software**, not a medical device and not a substitute for clinical judgment.

## Research thesis

Most medical language models are optimized to generate text. GAX studies a different question:

> Can a compact decision model select evidence-supported clinical actions, remain calibrated under uncertainty and distribution shift, and abstain when the available patient information is insufficient?

The project is designed around four first-class properties:

1. **Grounded actions** — decisions are scored against explicit candidate actions and supporting evidence.
2. **Calibrated uncertainty** — probabilities are evaluated with proper scoring rules, calibration error, and selective-risk metrics.
3. **Native abstention** — a high top-1 action score is not itself permission to act.
4. **FHIR-aware evaluation** — interoperability and EHR action selection are evaluated as core capabilities rather than demos.

## Planned research artifacts

- **GAX models** — compact open-weight health decision models.
- **GAXBench** — evaluation for clinical action selection, abstention, calibration, counterfactual sensitivity, evidence support, and FHIR workflows.
- **GAX training recipe** — reproducible data construction, hard-negative mining, counterfactual generation, calibration, and selective prediction.
- **GAX SDK** — typed Python API and a System-One-compatible HTTP surface.
- **FHIR adapter** — canonical rendering and action interfaces for FHIR resources.
- **Fine-tuning tutorials** — reproducible notebooks intended to run on accessible hardware where feasible.
- **Paper and reproducibility package** — manuscript, experiment manifests, raw metrics, environment locks, and figure/table generation.

## Scientific commitments

GAX will not claim that constrained outputs make clinical decisions correct. A model can be schema-valid and still be clinically wrong.

Public claims must be bound to reproducible evidence:

- exact model and dataset revisions;
- immutable train/validation/test splits;
- hardware and software environment;
- complete commands and random seeds;
- raw result artifacts;
- confidence intervals where appropriate;
- explicit failure cases and negative results.

No benchmark result should be added to the project summary until its evidence packet is reproducible.

## Status

**Research foundation phase.** Architecture, benchmark design, data governance, baselines, hypotheses, and experiment gates are being specified before implementation.

See the research plan and project governance documents as they land in this repository.

## License

The intended project code license is Apache-2.0. Dataset, model-weight, and third-party artifact licenses will be tracked independently and must be compatible with their actual sources before redistribution.
