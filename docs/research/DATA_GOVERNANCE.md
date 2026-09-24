# Data Governance and Provenance

GAX is a health-AI research project. Data governance is part of the scientific method, not a release checklist.

## Principles

1. Do not commit protected health information (PHI).
2. Do not redistribute data merely because it is downloadable.
3. Track dataset, revision, license, access terms, transformation code, and split lineage.
4. Keep train/dev/test boundaries at the originating entity level whenever possible.
5. Separate source truth from synthetic transformations.
6. Never use test labels to generate training examples, tune calibration, or select prompts.
7. Prefer reproducible public/synthetic builders when clinical data cannot be redistributed.

## Dataset registry

Every admitted source gets a machine-readable registry entry containing:

```yaml
id:
name:
source_url:
upstream_revision:
license:
redistribution:
access_requirements:
contains_clinical_text:
contains_phi:
patient_or_entity_key:
intended_uses:
forbidden_uses:
split_strategy:
transforms:
derived_artifacts:
review_status:
reviewer:
review_date:
```

A dataset may be referenced before approval but may not enter training/evaluation until `review_status=approved`.

## Restricted datasets

Examples such as credentialed clinical corpora may permit research access while prohibiting redistribution of raw records.

For such sources:

- store only scripts/config needed to build derived artifacts in an authorized environment;
- never upload raw records to GitHub or public Hugging Face repositories;
- do not publish recoverable text fragments;
- verify whether embeddings, labels, statistics, or model weights are considered derivatives under the source terms;
- provide a public synthetic alternative for smoke/reproduction tests where feasible.

## Synthetic FHIR

Synthetic data is preferred for public end-to-end FHIR fixtures when it provides the required signal.

Requirements:

- generation source and version recorded;
- deterministic seeds when possible;
- no accidental copying of real records;
- clinically impossible synthetic combinations tracked as either intentional adversarial cases or rejected;
- task labels generated from explicit logic and independently tested.

## Derived decision tasks

A derived task must preserve lineage:

```text
source record
  -> normalized record
  -> decision transform
  -> evidence transform
  -> counterfactual transform
  -> split artifact
```

Each stage receives a stable transformation revision.

## Counterfactual generation

Counterfactuals are high-risk because naive edits can create clinically incoherent records.

Rules:

- use a bounded edit taxonomy;
- preserve units and reference context;
- distinguish factual edits from label-only edits;
- run rule-based validity checks;
- clinician review a statistically meaningful sample before making clinical claims;
- publish the edit generator and failure/rejection counts;
- pair material edits with irrelevant controls.

## Evidence provenance

Evidence items must carry:

- source;
- source revision/date;
- local evidence ID;
- retrieval method if retrieved;
- passage boundaries or structured fact identifiers;
- support/contradict/irrelevant relation used by the benchmark;
- licensing status.

Do not treat model-generated rationale as ground-truth evidence.

## Split integrity

Order of operations:

1. identify patient/entity/source grouping;
2. assign split at group level;
3. generate derived examples inside each split;
4. run near-duplicate detection across splits;
5. freeze test membership;
6. generate and sign split manifest/checksums.

If a public benchmark has an official split, preserve it and document any additional leakage safeguards.

## Contamination

For every benchmark:

- check exact/near duplicate overlap with GAX training registries;
- check overlap between transformed variants;
- document known/pretraining contamination risk for base models;
- create task-family and source holdouts that are less vulnerable to memorization;
- avoid calling knowledge-benchmark performance a clinical validation.

## Human review

Where clinician/expert review is used, report:

- qualification definition;
- number of reviewers;
- instructions;
- blinded/randomized procedure where applicable;
- adjudication;
- inter-rater reliability where meaningful;
- conflicts of interest/compensation when relevant.

The project must not imply clinician review where none occurred.

## Release checklist

A public dataset/benchmark release requires:

- approved registry entries;
- license compatibility;
- no PHI leak;
- deterministic or documented builder;
- data/benchmark card;
- checksums;
- split manifest;
- provenance schema;
- known limitations;
- transformation tests;
- takedown/contact path.

## Model-weight release

Before releasing weights, review:

- base-model license;
- all training data obligations;
- whether restricted data terms permit model-weight release;
- required attribution/NOTICE;
- model card safety/intended-use language.

Code, datasets, and model weights can have different licenses. The repository's Apache-2.0 code license does not override upstream data/model terms.
