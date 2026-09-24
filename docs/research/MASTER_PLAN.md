# GAX Research Master Plan

Status: **FOUNDATION / pre-implementation**

This document is the governing research plan for GAX. It deliberately separates the research question from any preferred implementation. Architecture choices remain hypotheses until they survive the evaluation gates defined here.

## 1. Mission

GAX studies whether a compact non-generative decision model can serve as a reliable decision layer for health AI.

The target behavior is:

```text
clinical state + allowed actions + optional evidence
                    |
                    v
      typed probability distributions
                    |
       +------------+-------------+
       |            |             |
     action       support      sufficiency
       |            |             |
       +------------+-------------+
                    |
               act / abstain
```

GAX is not a diagnosis chatbot. It is not a replacement for a clinician. It is not allowed to emit an arbitrary clinical action outside the caller-provided schema.

## 2. Primary research question

**RQ0.** Can an evidence-aware non-generative health decision model improve the accuracy-calibration-selectivity trade-off for clinical action selection while retaining materially lower inference cost than autoregressive decision generation?

The paper is successful only if the answer is supported by controlled experiments. The project must publish null or negative results when a proposed mechanism does not help.

## 3. Research questions

- **RQ1 — Action selection:** Can GAX match or improve strong typed-decision and generative baselines on health decision tasks?
- **RQ2 — Calibration:** Are GAX probabilities better calibrated under in-domain and shifted conditions?
- **RQ3 — Abstention:** Does GAX reduce harmful overcommitment when critical information is absent, contradictory, or out of distribution?
- **RQ4 — Evidence grounding:** Does explicit evidence supervision improve action selection and support verification beyond ordinary classification?
- **RQ5 — Counterfactual sensitivity:** Does the model change its decision for clinically material fact changes while remaining stable to irrelevant edits?
- **RQ6 — Interoperability:** Can the same decision interface operate on FHIR-derived states and improve routing/action selection in EHR-agent workflows?
- **RQ7 — Efficiency:** What accuracy/calibration/selectivity is obtained per parameter, millisecond, memory footprint, and energy proxy?

## 4. Falsifiable hypotheses

- **H1:** domain-specific encoder representations improve health decision performance over similarly sized general encoders.
- **H2:** contrastive state-action training improves candidate ranking and action reuse efficiency over a plain classification head.
- **H3:** evidence-linked supervision improves evidence-support discrimination and reduces unsupported confident actions.
- **H4:** explicit insufficiency/abstention training improves selective risk at fixed coverage relative to confidence-threshold-only baselines.
- **H5:** clinically controlled counterfactual training increases causal sensitivity without increasing instability to semantically irrelevant perturbations.
- **H6:** a compact GAX model provides a better accuracy-latency-memory Pareto point than substantially larger autoregressive baselines for closed-action tasks.

Every hypothesis receives an ablation. No mechanism may be described as a contribution if it fails its predeclared gate.

## 5. Candidate model families

GAX will compare, not assume, three model families.

### A. Clinical encoder + typed heads

Initial backbone candidates:

- BioClinical ModernBERT base (~150M).
- BioClinical ModernBERT large (~396M).
- a general ModernBERT-sized control.

Advantages: bidirectional representations, long clinical context, accessible fine-tuning cost.

### B. Dual/shared encoder contrastive decision model

Inspired by the state-action separation demonstrated by CLM, but trained and evaluated independently for health.

Candidate design:

- shared or partially shared text encoder;
- state projection head;
- action projection head;
- normalized embeddings;
- temperature-scaled similarity;
- reusable/cached action embeddings;
- optional evidence representation.

### C. One-pass decoder/answer-slot baseline

A non-generative decoder reads state and typed options and uses restricted logits at decision slots. This family is represented by strong current open decision models and is required as a baseline.

The winning GAX architecture is selected by preregistered development metrics, not preference.

## 6. Working objective: ECAL

**Evidence-Calibrated Action Learning (ECAL)** is a working hypothesis, not a claim.

Candidate objective:

```text
L = w_action * L_action
  + w_evidence * L_evidence
  + w_cal * L_proper
  + w_abstain * L_selective
  + w_cf * L_counterfactual
  + w_replay * L_retention
```

Components:

- `L_action`: closed-action supervised or contrastive decision loss.
- `L_evidence`: positive/negative evidence support discrimination.
- `L_proper`: strictly proper probabilistic scoring objective (log loss and/or Brier-compatible formulation).
- `L_selective`: insufficiency and abstention objective.
- `L_counterfactual`: controlled sensitivity to clinically material changes.
- `L_retention`: replay to prevent collapse of broad decision competence.

Ablations remove each component independently and in interaction where justified.

## 7. Output contract

The research API should separate **belief over allowed actions** from **permission to act**.

```json
{
  "action": {
    "choice": "urgent_clinician_review",
    "probabilities": {
      "routine_followup": 0.08,
      "additional_test": 0.21,
      "medication_review": 0.17,
      "urgent_clinician_review": 0.54
    }
  },
  "evidence_support": 0.73,
  "information_sufficiency": 0.41,
  "abstain": true,
  "decision_policy": {
    "coverage_target": 0.80,
    "threshold_revision": "..."
  }
}
```

A top-1 action is never itself permission to execute a clinical action.

## 8. Benchmark policy

GAXBench is a suite, not a single aggregate score. It must include:

1. closed-action clinical knowledge;
2. evidence support / verifier tasks;
3. missing-information abstention;
4. contradiction abstention;
5. counterfactual sensitivity;
6. irrelevant-perturbation stability;
7. FHIR retrieval/routing/action selection;
8. specialty/source distribution shift;
9. calibration;
10. systems efficiency.

Primary metrics include accuracy/F1 where appropriate, NLL, Brier score, ECE with disclosed binning, AURC, risk at fixed coverage, abstention precision/recall, evidence AUROC/AUPRC, latency, peak memory, and throughput.

No single composite leaderboard score may hide regressions in safety-critical dimensions.

## 9. Baselines

At experiment freeze, include the strongest reproducible representatives available from these classes:

- TypeSafe Jev, if evaluation access and terms permit reproducible black-box measurement;
- Contrastive-LM/CLM;
- Laya;
- Mapika/decider;
- open one-pass/restricted-logit System-One implementations;
- plain BioClinical ModernBERT classifier/ranker;
- strong open-weight health/general LLM structured-output baseline;
- retrieval/evidence verifier baseline where relevant;
- trivial frequency/chance and heuristic controls.

Versions and revisions are frozen before final test evaluation.

## 10. Data program

Training data is divided into provenance-preserving stages:

- **D0 general decisions:** permitted public typed-decision data for learning the interface.
- **D1 biomedical decisions:** licensed biomedical/clinical classification and QA transformed into closed actions.
- **D2 evidence-linked decisions:** state/action/evidence triples with source identifiers.
- **D3 abstention:** incomplete, contradictory, corrupted, and OOD states with explicit insufficiency labels.
- **D4 counterfactuals:** controlled single-fact clinical edits with expected action-distribution changes.
- **D5 FHIR:** synthetic or permitted FHIR resources with deterministic task generation.
- **D6 agent traces:** only where provenance and redistribution rights are explicit.

Raw restricted clinical text must never be committed to the repository or republished through GAX artifacts.

## 11. Leakage controls

- split by patient/entity before derived examples are generated;
- split by source/task family for held-out generalization tests;
- hash near-duplicates across splits;
- maintain a transformation lineage ID;
- freeze test sets before model selection;
- no test-derived teacher data;
- audit benchmark overlap with training registries;
- record backbone-pretraining contamination risk when it cannot be ruled out.

## 12. Statistical protocol

- report point estimates plus 95% confidence intervals;
- use paired bootstrap for paired task comparisons where appropriate;
- predeclare the primary metric for each benchmark slice;
- adjust or clearly label exploratory multiple comparisons;
- report per-task distributions, not only macro averages;
- run multiple seeds for train-sensitive experiments;
- retain all valid runs, including regressions and negative results.

A superiority claim requires matched evaluation conditions and a confidence interval/test supporting the claim.

## 13. Compute strategy

The founder-cost constraint is **zero paid cloud spend**.

Research is designed in tiers:

- Tier 0: CPU unit tests and tiny fixtures.
- Tier 1: free Colab/Kaggle smoke and tutorial runs.
- Tier 2: free/sponsored GPU qualification for GAX-base.
- Tier 3: large-scale runs only on donated, sponsored, institutional, or otherwise zero-founder-cost compute.

Every run records hardware, wall time, peak memory, software lock, and whether compute was free/sponsored.

The paper must not depend on inaccessible infrastructure for its core reproducibility claim.

## 14. Safety and scope

GAX is research software. The initial release must not be presented for autonomous diagnosis, treatment, prescribing, or patient-facing emergency decision making.

Required controls:

- explicit research-use warning;
- abstention and human-escalation examples;
- model/data cards;
- intended-use and out-of-scope sections;
- subgroup and shift analysis where data supports it;
- failure-case appendix;
- no claims of "cannot hallucinate" or "clinically safe" without appropriate clinical evidence.

## 15. Publication plan

The first complete research package targets:

1. **arXiv preprint** with code, model card, benchmark card, and reproducibility bundle.
2. **peer-reviewed conference submission** selected when results and active calls are known. Candidate communities include MLHC/CHIL and major ML/NLP venues when the contribution fits.
3. **journal extension** only if the work gains sufficient clinical validation and additional analysis; candidate health-informatics/digital-medicine venues are evaluated at that time.

Venue selection must follow the actual contribution rather than optimizing the project toward a venue name.

## 16. Definition of paper-ready

The project is paper-ready only when all are true:

- benchmark schemas and frozen test splits are versioned;
- all baseline versions are frozen;
- at least one GAX model is reproducibly trainable;
- all primary tables regenerate from raw artifacts;
- ablations for all claimed mechanisms are complete;
- selective-risk and calibration evaluation is complete;
- FHIR evaluation is complete or explicitly removed from the claims;
- failure analysis is written;
- data/license audit is complete;
- model and benchmark cards are complete;
- manuscript contains no unsupported superlatives or hidden benchmark exclusions.

## 17. Stop conditions

A research direction is stopped or reframed if:

- improvement disappears under leakage-safe splits;
- calibration gains come only from post-hoc test tuning;
- abstention improves by refusing nearly everything;
- evidence gains do not survive evidence swaps;
- FHIR gains reduce to formatting artifacts;
- latency claims depend on incomparable hardware/protocols;
- a data license prevents reproducible publication.

The project should prefer a smaller defensible contribution over a larger unverifiable story.
