from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from gaxbench.external_adapters import (
    JSONCommandAdapter,
    TypeSafeHTTPAdapter,
    TypeSafeHTTPConfig,
)
from gaxbench.io import load_items
from gaxbench.schema import Action, BenchmarkItem, Evidence, Gold, Provenance

FIXTURES = Path(__file__).parent / "fixtures"


def evidence_item() -> BenchmarkItem:
    return BenchmarkItem(
        id="evidence-case",
        source_id="synthetic-evidence",
        split="test",
        task_family="transport",
        state={"signal": 1},
        actions=[
            Action(id="a", description="Action A"),
            Action(id="b", description="Action B"),
        ],
        gold=Gold(action="a", sufficient=True),
        evidence=[
            Evidence(
                id="e1",
                relation="support",
                text="Visible evidence text",
                source_ref="synthetic-source",
            )
        ],
        provenance=Provenance(
            dataset="synthetic",
            revision="1",
            license="CC0-1.0",
            transform_revision="1",
        ),
    )


@contextmanager
def fake_systemone_server(
    response: dict[str, Any],
    *,
    expected_authorization: str | None = None,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            if expected_authorization is not None:
                assert self.headers.get("Authorization") == expected_authorization
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            received.append(payload)
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_typesafe_http_adapter_hides_evidence_relation_and_normalizes_rounding() -> None:
    response = {
        "model": "synthetic-system-one",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.6,
                "probabilities": {"a": 0.6, "b": 0.3999},
            }
        },
    }
    with fake_systemone_server(response) as (base_url, received):
        adapter = TypeSafeHTTPAdapter(
            name="synthetic-typesafe",
            adapter_version="0.1",
            source_revision="source-sha",
            config=TypeSafeHTTPConfig(base_url=base_url),
            model_id="synthetic-model",
            model_revision="model-sha",
        )
        prediction = adapter.predict(evidence_item())

    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.metadata["transport_renormalized"] is True
    assert prediction.metadata["response_model"] == "synthetic-system-one"
    request = received[0]
    assert request["model"] == "synthetic-model"
    assert "gold" not in request
    visible_evidence = request["state"]["evidence"][0]
    assert visible_evidence["text"] == "Visible evidence text"
    assert "relation" not in visible_evidence


def test_typesafe_http_adapter_reads_api_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAXBENCH_TEST_API_KEY", "secret-value")
    response = {
        "answers": {
            "action": {
                "type": "choice",
                "choice": "routine",
                "confidence": 0.5,
                "probabilities": {"routine": 0.5, "urgent": 0.5},
            }
        }
    }
    with fake_systemone_server(
        response,
        expected_authorization="Bearer secret-value",
    ) as (base_url, _):
        adapter = TypeSafeHTTPAdapter(
            name="synthetic-typesafe",
            adapter_version="0.1",
            source_revision="source-sha",
            config=TypeSafeHTTPConfig(
                base_url=base_url,
                api_key_env="GAXBENCH_TEST_API_KEY",
            ),
        )
        prediction = adapter.predict(load_items(FIXTURES / "items.jsonl")[0])
    assert prediction.item_id == "case-1"


def test_typesafe_http_adapter_rejects_action_key_mismatch() -> None:
    response = {
        "answers": {
            "action": {
                "type": "choice",
                "choice": "wrong",
                "confidence": 1.0,
                "probabilities": {"wrong": 1.0},
            }
        }
    }
    with fake_systemone_server(response) as (base_url, _):
        adapter = TypeSafeHTTPAdapter(
            name="synthetic-typesafe",
            adapter_version="0.1",
            source_revision="source-sha",
            config=TypeSafeHTTPConfig(base_url=base_url),
        )
        with pytest.raises(ValueError, match="keys mismatch"):
            adapter.predict(load_items(FIXTURES / "items.jsonl")[0])


def test_json_command_adapter_uses_stdin_without_label_leakage() -> None:
    script = """
import json, sys
request = json.loads(sys.stdin.read())
assert "gold" not in request
assert "relation" not in json.dumps(request["state"])
actions = request["actions"]
p = 1.0 / len(actions)
print(json.dumps({
    "item_id": request["item_id"],
    "probabilities": {action["id"]: p for action in actions},
}))
"""
    adapter = JSONCommandAdapter(
        [sys.executable, "-c", script],
        name="synthetic-command",
        adapter_version="0.1",
        source_revision="source-sha",
        deterministic=True,
    )
    prediction = adapter.predict(evidence_item())
    assert prediction.item_id == "evidence-case"
    assert prediction.metadata["transport"] == "json-command-v1"
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_json_command_adapter_surfaces_nonzero_exit() -> None:
    adapter = JSONCommandAdapter(
        [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"],
        name="failing-command",
        adapter_version="0.1",
        source_revision="source-sha",
        deterministic=True,
    )
    with pytest.raises(RuntimeError, match="exited with 3"):
        adapter.predict(evidence_item())
