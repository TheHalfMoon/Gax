# P08 Full Paper Evaluation Protocol

Status: **PRE-TEST / final-test access sealed**

P08 is the first GAX phase allowed to produce paper-eligible comparative results. It is also the phase with the greatest risk of accidental test tuning, benchmark drift, selective reporting, and hardware-incomparable claims. This protocol therefore treats final-test access as an explicit, digest-bound state transition rather than an informal researcher decision.

## 1. Research objective

Evaluate whether a compact, non-generative health decision model can improve the accuracy-calibration-selectivity trade-off for bounded clinical decisions while providing evidence-sensitive behavior, explicit information sufficiency, and FHIR-aware action selection.

The paper is not required to prove every proposed mechanism. P08 must preserve null, negative, blocked, timeout, OOM, parsing, and access-limited outcomes.

## 2. Novelty boundary

The open System-One / typed-decision ecosystem expanded rapidly after Jev's September 2026 launch. General claims such as "non-generative typed decisions are fast" or "a compact encoder can return option probabilities" are no longer sufficient novelty for GAX.

The intended paper contribution must survive at the health-specific intersection of:

```text
typed probabilistic decisions
        x
clinical-domain representations
        x
evidence intervention / support verification
        x
information sufficiency and selective abstention
        x
FHIR-aware resource/action selection
```

A literature refresh is mandatory before the final claim freeze. If concurrent work closes a gap, the corresponding GAX claim must be narrowed, downgraded to related work, or removed.

## 3. Final-test state machine

P08 has exactly two final-test states:

- `sealed`: development, calibration, protocol authoring, adapter qualification, and audit work may continue; final-test labels/results may not be used.
- `authorized`: the complete freeze manifest passed validation and has a digest bound to the exact frozen protocol.

The transition is implemented by `gax-p08 freeze-authorize`. The authorized artifact must be written to a separate path from the sealed source manifest.

Changing any output-affecting frozen field after authorization creates a new experiment revision and requires a complete affected rerun.

## 4. Required freeze surfaces

The P08 freeze manifest binds:

- repository revision and clean-tree state;
- P07 canonical closeout evidence;
- benchmark/dataset revisions;
- immutable split-manifest hashes;
- leakage-audit hashes;
- license-audit hashes and redistribution state;
- GAX and baseline identities;
- adapter revisions;
- training seeds where applicable;
- calibration method and calibration split;
- target coverage grid;
- statistical analysis plan;
- hardware protocol;
- failure policy;
- evidence/intervention protocol;
- FHIR representation revision.

Authorization fails when the repository tree is dirty, a benchmark test manifest is not frozen, a required system is neither qualified nor explicitly blocked, or the digest does not match the frozen payload.

## 5. Canonical comparison classes

Use `registry/baselines.json` as the identity source of truth.

Required comparison classes include, where reproducibly executable:

- GAX deterministic controls;
- final selected GAX checkpoint(s);
- CLM;
- Laya;
- decider;
- restricted-logit / no-task-training control;
- BioClinical ModernBERT clinical-encoder control;
- Qwen3.5-4B structured-output generative control;
- TypeSafe Jev only when reproducible access and terms permit matched measurement.

A blocked baseline stays visible in the registry, claim ledger, and limitations. It must not be silently removed to improve the story.

## 6. Benchmark families

Do not reduce GAXBench to one aggregate number.

### 6.1 Bounded clinical decisions

Measure closed-action task quality and probability quality. These results do not by themselves establish clinical safety.

### 6.2 Information insufficiency and abstention

Measure missing information, contradiction, out-of-scope requests, impossible requests, and permitted external abstention benchmarks.

### 6.3 Evidence grounding

Use controlled support, removal, irrelevance, swap, and contradiction interventions. A grounding claim requires the model's beliefs to respond to evidence changes in the expected direction while remaining stable to irrelevant controls.

### 6.4 Counterfactual robustness

Evaluate clinically material paired edits and matched irrelevant edits. Report directional response and stability. Do not label the result causal without an identification design that justifies that term.

### 6.5 FHIR / EHR-agent decisions

Preserve source-native FHIR version identity and separate:

- resource-type selection;
- read/search routing;
- information-sufficiency / stopping;
- support verification;
- retrieval quality where upstream gold labels permit it;
- full generative-agent task success when evaluated externally.

GAX's read-only P07 boundary remains in force.

### 6.6 Distribution shift

Report source, specialty, note-style, coding-vocabulary, context-length, and ethical subgroup slices only where the source data supports valid analysis.

## 7. Primary metrics

Applicable paper slices must report exact requested/completed/failed counts and include:

- accuracy and macro-F1 where valid;
- NLL;
- multiclass Brier score;
- disclosed-bin ECE and reliability diagrams;
- full risk-coverage curve;
- AURC;
- risk@50, risk@80, and risk@90 coverage;
- actual policy coverage and committed risk;
- abstention precision/recall/F1;
- unsafe-commit and over-abstain rates;
- information-sufficiency calibration;
- evidence AUROC/AUPRC where the label structure makes them valid;
- intervention probability deltas;
- counterfactual directional sensitivity and irrelevant-edit stability;
- FHIR routing and unsupported-action rates;
- p50/p95 latency, throughput, and peak memory under the frozen systems protocol.

`risk@80` is a primary selective-prediction metric candidate, not a replacement for ordinary accuracy or the full risk-coverage curve.

## 8. Statistical protocol

Primary comparisons are paired on identical eligible items.

Required:

- 95% confidence intervals;
- paired bootstrap where appropriate;
- frozen bootstrap seed and replicate count;
- multiple training seeds for train-sensitive mechanisms;
- declared primary comparisons before final-test authorization;
- explicit multiple-comparison policy;
- per-task/per-slice reporting;
- no silent denominator changes.

A superiority claim requires matched conditions plus uncertainty evidence that supports the claim.

## 9. Calibration and selective prediction

Calibration parameters and abstention/selective thresholds are fit on calibration/development data only.

Report raw and calibrated distributions separately. Learned information sufficiency must be compared with max-probability, entropy, and margin controls at matched coverage. A method does not win merely by refusing most examples.

## 10. ECAL decision gate

Every proposed ECAL component remains `keep`, `reject`, or `blocked` based on frozen development evidence before final-test opening:

- state/action contrastive objective;
- hard negatives;
- evidence objective;
- proper-scoring/calibration objective;
- replay/retention;
- preregistered interactions.

A mechanism that fails its development gate remains in the experiment ledger and cannot re-enter after observing final-test outcomes without a new experiment revision.

## 11. Failure accounting

Paper-eligible runs must record:

- requested;
- completed;
- timeout;
- OOM;
- transport failure;
- parse/interface failure;
- invalid probability/schema output;
- preregistered exclusion with reason.

Wrong in-schema answers count as wrong. Failed inference is never silently dropped.

## 12. Efficiency protocol

Direct speed claims require matched hardware and disclosed:

- batch/concurrency;
- warm/cold state;
- context-length distribution;
- candidate/question count;
- preprocessing inclusion;
- cache hit/miss state;
- runtime/driver versions;
- p50/p95 latency;
- throughput and peak memory.

When hardware differs, report absolute measurements separately and do not claim direct speed superiority.

## 13. Claim ledger

Every candidate public claim has a machine-readable status:

- `candidate`;
- `supported`;
- `null`;
- `rejected`;
- `blocked`;
- `exploratory`.

Only `supported` claims with evidence-packet identifiers are exportable as affirmative public claims. Null, rejected, and blocked entries require reasons and remain visible. Exploratory claims cannot be primary candidates.

The claim ledger is not a marketing checklist. It is a mechanism for preventing unsupported superlatives from entering the abstract, README, paper, or release notes.

## 14. Artifact generation

Every paper row must bind the existing GAX evidence-packet contract. Tables and figures must be generated from raw versioned artifacts, not manually transcribed numbers.

Each generated artifact records its source run IDs or hashes.

## 15. Compute boundary

Founder-paid cloud compute is prohibited.

Use local/free Colab/Kaggle resources where feasible and donated, sponsored, or institutional compute for runs that exceed those resources. Compute unavailability is recorded as `blocked`; it is not permission to substitute an incomparable result.

## 16. Safety boundary

P08 is research evaluation. It does not authorize autonomous diagnosis, prescribing, treatment, triage, emergency execution, EHR writes, clinical-safety claims, regulatory-readiness claims, or FHIR conformance certification.

## 17. P08 freeze-grain exit

The initial P08 freeze grain is complete only when:

- final-test access is code-enforced and sealed by default;
- authorization is digest-bound;
- blocked systems require explicit reasons;
- benchmark split/license/leakage hashes are mandatory;
- statistical and protocol freezes are represented;
- claim states are machine-readable and supported claims require evidence IDs;
- governance CLI paths are tested;
- synthetic fixtures contain no patient data or paper performance result;
- cross-platform CI passes on the exact head;
- the freeze-grain PR merges with an expected-head guard and post-main CI succeeds.

This grain does **not** authorize final-test access by itself. A real P08 freeze manifest populated with audited data/model identities must later pass authorization.
