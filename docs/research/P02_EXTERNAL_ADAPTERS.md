# P02 External Adapter Qualification

Status: **implementation grain SG-000004**

This grain extends the canonical P02 harness with transports for external baseline
implementations while preserving the P01 metric and failure contracts.

## Scope

The grain adds two transport adapters:

- `TypeSafeHTTPAdapter` for implementations exposing `POST /v1/systemone`;
- `JSONCommandAdapter` for local executables that emit a typed GAXBench prediction.

The HTTP transport is directly applicable to the currently frozen CLM, Laya, and
decider source revisions. The command transport is a process boundary for open
baselines that do not expose the shared HTTP protocol.

This grain qualifies **transport behavior**, not real-model benchmark performance.

## No-label-leak rendering

A benchmark item contains both model-visible content and annotations used by
evaluation. External adapters must never receive:

- `gold`;
- `evidence.relation`;
- provenance labels that encode benchmark construction;
- sufficiency labels.

When evidence is present, only its ID, text or structured content, and source
reference are sent to the model. The relation label stays evaluator-only.

## TypeSafe-compatible request

Each item becomes one Choice question:

```json
{
  "state": "<model-visible state>",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Select the best allowed action for the provided state.",
      "criteria": {
        "<action-id>": "<action description>"
      }
    }
  }
}
```

The adapter consumes only the returned `probabilities` mapping. It does not
trust the server's top-choice field as a substitute for the full distribution.

## Transport rounding

Some compatible servers serialize probabilities to four decimal places. The P01
kernel requires normalized distributions within `1e-6`.

The HTTP adapter therefore permits only the maximum error explainable by
four-decimal per-action rounding:

```text
allowed absolute sum error = number_of_actions * 5e-5 + 1e-9
```

A distribution inside that bound is renormalized and records:

- `raw_probability_sum`;
- `transport_renormalized=true`.

Larger normalization errors fail the item. This is a transport correction, not
model calibration.

## Secret handling

API secrets are referenced by environment-variable name. The secret value never
appears in the CLI arguments or evidence packet command.

## Frozen upstream sources

At authoring time the following public source revisions were frozen for adapter
qualification:

- CLM: `7956937c58ed5839c06ddc4dc6b6b61c3a3e4094` — Apache-2.0;
- Laya: `3c68ca2ccf6a83640ab80c20379503fe72c772fd` — Apache-2.0;
- decider: `5f91c011f05fa4b685f0845281a0805a56eb0169` — Apache-2.0;
- open-alternative-jev: `4a85df1831b537343c9e133b5150fa3a3b1ce98e` — Apache-2.0.

CLM, Laya, and decider publicly expose the TypeSafe-compatible
`POST /v1/systemone` request shape at these source revisions.

The restricted-logit project does not expose that HTTP server. Its source and
license are frozen, but a real model run remains external-pending and can use
the JSON-command or prediction-file transport after an immutable model revision
is chosen.

## Qualification boundary

`adapter-qualified` means:

- the source and license are frozen;
- GAX's protocol adapter is covered by deterministic offline tests;
- action-key mismatch, malformed output, non-zero process exit, and label leakage
  are fail-closed;
- evidence packets preserve source/model revision fields.

It does **not** mean the upstream model has been executed on GAXBench.

Real model qualification requires a separate zero-founder-cost run with immutable
model/tokenizer revisions and a complete evidence packet.

## CLI examples

TypeSafe-compatible server:

```bash
gaxbench baseline-run \
  --items <items.jsonl> \
  --adapter typesafe-http \
  --base-url http://127.0.0.1:8700 \
  --adapter-name clm \
  --source-revision 7956937c58ed5839c06ddc4dc6b6b61c3a3e4094 \
  --model-id <model-id> \
  --model-revision <immutable-model-revision> \
  --output-dir <packet-dir> \
  --repo-revision <gax-sha>
```

Local command adapter:

```bash
gaxbench baseline-run \
  --items <items.jsonl> \
  --adapter json-command \
  --command-json '["python","adapter.py"]' \
  --adapter-name <baseline-name> \
  --source-revision <source-sha> \
  --output-dir <packet-dir> \
  --repo-revision <gax-sha>
```

## Exit gate

SG-000004 is PROVEN only after exact-head CI passes Ruff, mypy strict, pytest,
and compileall on Python 3.11/3.12 across Linux and Windows, followed by normal
merge and successful post-merge CI.
