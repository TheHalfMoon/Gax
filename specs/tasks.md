# GAX Program Tasks

## P00 — Research foundation

- [x] Initialize public repository
- [x] Research thesis
- [x] Literature/competitor map
- [x] Architecture search space
- [x] GAXBench specification
- [x] Data governance
- [x] Reproducibility and claim discipline
- [x] Publication plan
- [x] Roadmap
- [x] SpecGrain SG-000001
- [x] Foundation PR reviewed
- [x] Foundation PR merged
- [x] Canonical P00 closeout

## P01 — Benchmark kernel

- [x] Benchmark schema
- [x] Metrics implementation
- [x] Calibration metrics
- [x] Selective-risk metrics
- [x] Provenance registry
- [x] Synthetic fixtures
- [x] Duplicate/split audit
- [x] Cross-platform tests

## P02 — Baseline harness

- [x] evidence-packet runner
- [x] Shared TypeSafe HTTP external transport
- [x] Shell-free JSON command external transport
- [x] CLM source/license/protocol adapter qualification
- [x] Laya source/license/protocol adapter qualification
- [x] decider source/license/protocol adapter qualification
- [x] restricted-logit source/license freeze and canonical transport path
- [x] Clinical encoder identity/revision/license freeze
- [x] Structured-output LLM identity/revision/license freeze
- [x] Canonical P02 phase closeout

## P03 — GAX v0

- [x] Model interface
- [x] Training pipeline
- [x] Checkpoint format
- [x] Deterministic inference
- [x] GAX-base smoke model
- [x] Model card draft
- [x] Model-visible evidence boundary correction
- [x] Canonical P03 phase closeout

## P04 — ECAL research

- [x] Bidirectional/multi-positive state-action contrastive ablation
- [x] Hard-negative ablation
- [x] Evidence objective ablation
- [x] Proper-scoring/calibration ablation
- [x] Replay/retention ablation
- [x] Matched-ablation manifest and keep/reject ledger
- [x] Explicit experiment-context binding
- [x] Zero-norm feature-hash robustness regression
- [x] Canonical P04 phase closeout

## P05 — Abstention

- [x] Confidence baseline
- [x] Entropy baseline
- [x] Top-1/top-2 margin baseline
- [x] Learned information sufficiency
- [x] Calibration/selective-risk control
- [x] Matched-coverage evaluation
- [x] Unsafe-commit and over-abstention analysis
- [x] Keep/reject/defer ledger
- [x] Final-test protection
- [x] Exact-head Linux/Windows Python 3.11/3.12 qualification
- [x] Guarded implementation merge and post-main CI
- [x] Canonical P05 phase closeout

## P06 — Evidence and counterfactuals

- [x] Versioned intervention/pair schema
- [x] Source/intervention lineage contract
- [x] Counterfactual and evidence-intervention taxonomy
- [x] Controlled material-change pair representation
- [x] Irrelevant-edit control representation
- [x] Evidence support/removal/contradiction intervention representation
- [x] Same-split pair-family enforcement
- [x] Pair-conditioned robust-accuracy and counterfactual-failure metrics
- [x] Directional probability-shift metrics
- [x] Irrelevant-edit stability metrics
- [x] Evidence-support and information-sufficiency response metrics
- [x] Exact item/prediction denominator preservation
- [x] Abstract CC0 synthetic fixtures
- [x] Keep/reject/defer decision ledger
- [x] CLI and exact run-reproduction manifest
- [x] Final-test rejection during P06 mechanism qualification
- [x] Exact-head Linux/Windows Python 3.11/3.12 qualification
- [x] Guarded implementation merge and post-main CI
- [x] Canonical P06 phase closeout

P06 deliberately does not claim a generic medical transformation builder. Its proven surface is a governed pair manifest/evaluator plus abstract fixtures. Real medical intervention construction and paper-level mechanism decisions remain gated on licensed, reviewed, leakage-audited development evidence.

## P07 — FHIR

- [x] Stable FHIR R5 5.0.0 canonical / explicit R4 compatibility contract
- [x] Deterministic FHIR canonicalizer and Bundle/contained-resource checks
- [x] Read-only FHIR action schemas
- [x] Abstract CC0 FHIR fixtures and registry entry
- [x] Narrative/source-order/flat-text representation controls
- [x] MedAgentBench source/license/revision freeze
- [x] FHIR-AgentBench source/license/revision and R4 freeze
- [x] MedAgentBench local normalized-export adapter protocol
- [x] FHIR-AgentBench local normalized-export adapter protocol
- [x] Information-sufficiency/tool-routing reuse of GAXBench metrics
- [x] P07 final-test rejection in conversion/export paths
- [x] Exact-head cross-platform qualification
- [x] Guarded implementation merge and post-main CI
- [x] Canonical P07 closeout

P07 proves deterministic read-only interoperability mechanics and frozen source/adapter contracts. It does not prove clinical correctness, FHIR conformance certification, semantic R4-to-R5 conversion, or benchmark superiority. Those paper claims remain gated on P08.

## P08 — Full paper evaluation

### Freeze before final-test access

- [ ] Freeze benchmark version and immutable split manifests
- [ ] Freeze GAX model/checkpoint revisions and training seeds
- [ ] Freeze all baseline model/tokenizer/source revisions
- [ ] Freeze calibration method, calibration split, and selective policy
- [ ] Freeze ECAL component keep/reject candidate set from development evidence
- [ ] Freeze FHIR representation candidates and selection protocol
- [ ] Freeze hardware/timing protocol and comparability rules
- [ ] Complete dataset/license/redistribution audit
- [ ] Complete train/dev/calibration/test leakage audit
- [ ] Sign final-test opening manifest

### Real-model qualification

- [ ] Clinical encoder real-model qualification
- [ ] CLM real-model qualification
- [ ] Laya real-model qualification
- [ ] decider real-model qualification
- [ ] restricted-logit real-model qualification
- [ ] structured-output LLM real-model qualification
- [ ] Jev evaluation if reproducible access and terms permit; otherwise record explicit blocked status

### Primary paper evaluation

- [ ] Main action-selection tables
- [ ] NLL, Brier, ECE, reliability analysis
- [ ] Risk-coverage curves, AURC, and risk@50/80/90
- [ ] Matched-coverage abstention analysis with unsafe-commit and over-abstention rates
- [ ] ECAL component ablations with multiple seeds where train-sensitive
- [ ] Evidence-intervention evaluation
- [ ] Counterfactual material-sensitivity and irrelevant-edit stability evaluation
- [ ] FHIR representation and EHR-agent action-selection evaluation
- [ ] Distribution-shift slices
- [ ] Failure taxonomy and qualitative error analysis
- [ ] Latency, throughput, peak-memory, and candidate/context scaling
- [ ] Paired bootstrap confidence intervals and declared primary comparisons
- [ ] Preserve null, negative, timeout, OOM, and parse-failure outcomes

### Reproducibility and claim freeze

- [ ] Complete evidence packet for every paper-table row
- [ ] Generate every table from raw versioned artifacts
- [ ] Generate every figure from raw versioned artifacts
- [ ] Create paper claim ledger mapping each claim to exact evidence or rejection
- [ ] Freeze P08 final results without post-test tuning
- [ ] Canonical P08 closeout

## P09 — ArXiv and release

- [ ] Paper manuscript
- [ ] Appendix
- [ ] Reproducibility bundle
- [ ] GAXBench release
- [ ] Model release
- [ ] Fine-tuning notebooks
- [ ] Hugging Face release
- [ ] arXiv submission package
- [ ] Public release tag

## P10 — Peer review / external validation

- [ ] Independent reproduction
- [ ] Active venue CFP review
- [ ] Peer-reviewed submission
- [ ] Reviewer response artifacts
- [ ] Journal extension decision
