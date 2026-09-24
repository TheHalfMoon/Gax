# Literature and Competitor Map

Snapshot date: **2026-09-24**

This map is a research-positioning document, not an endorsement or ranking. Claims below should be re-verified at experiment freeze because this area is moving quickly.

## 1. System-One / typed-decision systems

### TypeSafe Jev

Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev

TypeSafe introduced Jev publicly on 2026-09-15 as a System One model: unstructured state in, typed probabilistic decisions out. The launch describes a training approach named Reinforcement Learning for Calibrated Decisions (RLCD).

What GAX should learn:

- probability distributions are a first-class API, not prose parsed after generation;
- typed decisions can be treated as a model class rather than only an application wrapper;
- calibration must be part of model quality.

What remains open for GAX:

- health-specific evidence grounding;
- selective abstention under missing/contradictory information;
- FHIR/EHR task evaluation;
- public, peer-review-ready ablations and training evidence.

Jev is a baseline only where access, terms, and reproducibility permit. GAX must not infer unpublished implementation details.

### Contrastive Language Models (CLM)

Repository: https://github.com/Contrastive-LM/CLM

CLM connects states and actions with contrastive learning and separates their representations so action embeddings can be cached. The public repository describes pre-training on broad QA pairs, hard-negative mid-training, and agentic post-training.

What GAX should learn:

- state/action disaggregation;
- hard-negative mining;
- action caching;
- contrastive ranking as a natural primitive for closed action sets;
- explicit fine-tuning and reproducibility artifacts.

GAX research gap:

- CLM is general-purpose rather than clinically grounded;
- evidence support and information sufficiency are not the central outputs;
- clinical counterfactuals and FHIR action selection remain separate research questions.

### Laya

Repository: https://github.com/NandhaKishorM/laya

Laya demonstrates an open, multilingual, non-autoregressive typed-decision system using encoder models, proper-scoring-focused training, and accessible fine-tuning workflows.

What GAX should learn:

- compact encoders can be serious decision models;
- fine-tuning should be accessible rather than requiring frontier-scale compute;
- probability calibration must be measured explicitly;
- multilingual and long-context behavior should be considered independently.

### decider

Repository: https://github.com/Mapika/decider

decider is an open family of one-pass typed-decision models based on Qwen3.5. It includes supervised training, calibration-aware RL experiments, multiple model scales, answer-slot probability readout, and explicit calibration/selective metrics.

What GAX should learn:

- strong answer-slot decoder baselines are mandatory;
- AURC and probability metrics belong beside accuracy;
- model history and negative/regressive releases should be documented;
- candidate-set and serving behavior need dedicated fidelity tests.

### Open Alternative to Jev

Repository: https://github.com/ikermoel/open-alternative-jev

This work shows that typed distributions can also be extracted from open-weight LLMs without task-specific training by restricting next-token distributions and packing questions.

Role in GAX:

- zero-shot restricted-logit baseline;
- evidence that architecture/training gains must beat a surprisingly strong no-training control;
- reminder to distinguish structural speedups from batching/padding artifacts.

## 2. Clinical encoders

### BioClinical ModernBERT

Paper: https://arxiv.org/abs/2506.10896

BioClinical ModernBERT adapts ModernBERT to biomedical and clinical text through continued pretraining on more than 53.5B tokens and releases approximately 150M and 396M parameter variants.

Why it is a primary GAX candidate:

- bidirectional encoder suited to discriminative tasks;
- compact enough for accessible fine-tuning;
- long-context clinical specialization;
- useful matched-size base/large experiment.

### Clinical ModernBERT

Paper: https://arxiv.org/abs/2504.03964

Clinical ModernBERT is another long-context clinical encoder incorporating biomedical literature, clinical notes, and medical ontologies.

Role in GAX:

- clinical-encoder baseline/candidate depending licensing and reproducibility;
- evidence that long-context encoder modeling is a serious alternative to decoder-only medical models.

## 3. Medical agents and FHIR

### MedAgentBench

Paper: https://arxiv.org/abs/2501.14654
Repository: https://github.com/stanfordmlgroup/MedAgentBench

MedAgentBench provides a realistic FHIR-compliant virtual EHR environment with 300 physician-authored patient-specific tasks, 100 patient profiles, and a large underlying record environment.

GAX use:

- agent action-selection evaluation;
- next-tool/next-resource routing tasks;
- information-sufficiency and abstention tasks;
- interoperability validation.

### FHIR-AgentBench

Paper: https://proceedings.mlr.press/v297/lee26a.html
Repository: https://github.com/glee4810/FHIR-AgentBench

FHIR-AgentBench grounds 2,931 real-world clinical questions in HL7 FHIR and compares retrieval/tool/reasoning strategies.

GAX use:

- FHIR retrieval/routing decision evaluation;
- canonicalization controls;
- selective stopping when enough information has been retrieved;
- comparison with ordinary generative agent approaches.

## 4. Medical verification and evidence grounding

### Med-PRM

Paper: https://aclanthology.org/2025.emnlp-main.837/

Med-PRM uses retrieved guidelines and literature to verify intermediate medical reasoning steps and reports gains when used as a process reward model.

GAX research implication:

- evidence-aware verification can improve medical reasoning systems;
- GAX should test evidence support directly rather than infer it from action confidence;
- evidence intervention experiments are necessary to show grounding rather than correlation.

GAX is not a process-reasoning generator; Med-PRM is therefore a neighboring verifier baseline, not an architectural equivalent.

## 5. Medical abstention

### MedQAbstain

Paper: https://aclanthology.org/2026.acl-long.1365/

MedQAbstain explicitly evaluates abstention under medical uncertainty and reports systematic overcommitment by state-of-the-art LLMs across its settings.

GAX research implication:

- abstention is not an optional UI feature;
- missing-information tasks must be part of training and evaluation;
- self-reported verbal confidence is insufficient;
- selective risk at fixed coverage should be a headline metric.

## 6. Calibration and selective prediction foundations

### Neural network calibration

Guo et al., *On Calibration of Modern Neural Networks*
https://arxiv.org/abs/1706.04599

Relevance:

- temperature scaling baseline;
- calibration error reporting;
- separation between predictive accuracy and probability quality.

### Selective classification

Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks*
https://arxiv.org/abs/1705.08500

Relevance:

- risk-coverage evaluation;
- principled comparison of abstaining predictors.

### Proper scoring rules

Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*
https://doi.org/10.1198/016214506000001437

Relevance:

- probability training/evaluation should reward honest distributions rather than only top-1 correctness.

## 7. Positioning statement

The intended GAX contribution is the intersection of:

```text
typed decision models
        x
clinical domain encoders
        x
evidence-aware verification
        x
selective abstention
        x
FHIR/agent action selection
```

No single cited work is assumed to leave this exact gap open forever. Before submission, the literature search must be repeated and the novelty statement updated against any work published after this snapshot.

## 8. Pre-submission literature refresh

Before freezing the paper:

- rerun searches for "System One medical model", "clinical decision model calibrated", "medical abstention LLM", "FHIR decision model", "medical verifier", and "clinical selective prediction";
- search arXiv, ACL Anthology, PMLR, PubMed, Google Scholar/Semantic Scholar where accessible;
- inspect citations to and from MedQAbstain, Med-PRM, MedAgentBench, FHIR-AgentBench, and BioClinical ModernBERT;
- update this file with publication dates, peer-review status, licenses, and overlapping claims;
- modify the paper contribution statement if a concurrent work closes part of the gap.
