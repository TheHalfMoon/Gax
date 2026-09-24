from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from gaxbench.baselines import AdapterIdentity
from gaxbench.schema import BenchmarkItem, Prediction


_ACTION_QUESTION_ID = "action"
_ACTION_INSTRUCTIONS = "Select the best allowed action for the provided state."
_MAX_ERROR_BODY_BYTES = 4096
_DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class TypeSafeHTTPConfig:
    base_url: str
    timeout_seconds: float = 300.0
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query or fragment components")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must be non-empty when provided")


class TypeSafeHTTPAdapter:
    """Adapter for public System-One servers exposing POST /v1/systemone.

    This transport is shared by CLM, Laya, and decider. It sends only model-visible
    benchmark content. Evidence relation labels and all gold fields are deliberately
    excluded from the request.
    """

    def __init__(
        self,
        *,
        name: str,
        adapter_version: str,
        source_revision: str,
        config: TypeSafeHTTPConfig,
        model_id: str | None = None,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        deterministic: bool = False,
    ) -> None:
        if not name:
            raise ValueError("name must not be empty")
        if not adapter_version:
            raise ValueError("adapter_version must not be empty")
        if not source_revision:
            raise ValueError("source_revision must not be empty")
        self._config = config
        self._identity = AdapterIdentity(
            name=name,
            adapter_version=adapter_version,
            deterministic=deterministic,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            source_revision=source_revision,
        )

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        if not items:
            raise ValueError("items must not be empty")
        for item in items:
            if not item.actions:
                raise ValueError(f"item {item.id!r} has no actions")

    def predict(self, item: BenchmarkItem) -> Prediction:
        payload = {
            "state": render_model_state(item),
            "questions": {
                _ACTION_QUESTION_ID: {
                    "type": "choice",
                    "instructions": _ACTION_INSTRUCTIONS,
                    "criteria": {
                        action.id: action.description for action in item.actions
                    },
                }
            },
        }
        if self._identity.model_id is not None:
            payload["model"] = self._identity.model_id

        response = self._post(payload)
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("System-One response must contain an answers object")
        answer = answers.get(_ACTION_QUESTION_ID)
        if not isinstance(answer, dict):
            raise ValueError("System-One response is missing the action answer")
        if answer.get("type") != "choice":
            raise ValueError("System-One action answer must have type='choice'")

        raw_probabilities = answer.get("probabilities")
        if not isinstance(raw_probabilities, dict):
            raise ValueError("System-One choice answer must contain probabilities")
        probabilities = _coerce_and_normalize_probabilities(
            raw_probabilities,
            expected_ids=[action.id for action in item.actions],
        )

        raw_sum = sum(float(value) for value in raw_probabilities.values())
        metadata: dict[str, Any] = {
            "transport": "typesafe-http-v1",
            "raw_probability_sum": raw_sum,
            "transport_renormalized": abs(raw_sum - 1.0) > 1e-6,
        }
        response_model = response.get("model")
        if isinstance(response_model, str):
            metadata["response_model"] = response_model

        return Prediction(
            item_id=item.id,
            probabilities=probabilities,
            metadata=metadata,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._config.api_key_env is not None:
            api_key = os.environ.get(self._config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"environment variable {self._config.api_key_env!r} is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"

        request = Request(
            self._config.base_url.rstrip("/") + "/v1/systemone",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._config.timeout_seconds) as response:
                raw = response.read(self._config.max_response_bytes + 1)
        except HTTPError as exc:
            detail = exc.read(_MAX_ERROR_BODY_BYTES).decode("utf-8", errors="replace")
            raise RuntimeError(f"System-One HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"System-One transport error: {exc.reason}") from exc

        if len(raw) > self._config.max_response_bytes:
            raise ValueError("System-One response exceeds max_response_bytes")
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("System-One response must be valid UTF-8 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("System-One response must be a JSON object")
        return parsed


class JSONCommandAdapter:
    """Run a local external adapter process once per benchmark item.

    The command receives one JSON request on stdin and must emit one Prediction-shaped
    JSON object on stdout. No shell is used. This transport is intended for baselines
    that do not expose the TypeSafe HTTP protocol.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str,
        adapter_version: str,
        source_revision: str,
        model_id: str | None = None,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        deterministic: bool,
        timeout_seconds: float = 300.0,
        max_stdout_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if not command or any(not part for part in command):
            raise ValueError("command must contain non-empty argv entries")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_stdout_bytes <= 0:
            raise ValueError("max_stdout_bytes must be positive")
        if not source_revision:
            raise ValueError("source_revision must not be empty")
        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds
        self._max_stdout_bytes = max_stdout_bytes
        self._identity = AdapterIdentity(
            name=name,
            adapter_version=adapter_version,
            deterministic=deterministic,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            source_revision=source_revision,
        )

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    def prepare(self, items: Sequence[BenchmarkItem]) -> None:
        if not items:
            raise ValueError("items must not be empty")

    def predict(self, item: BenchmarkItem) -> Prediction:
        request_payload = {
            "item_id": item.id,
            "state": render_model_state(item),
            "actions": [
                {"id": action.id, "description": action.description}
                for action in item.actions
            ],
        }
        serialized = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        try:
            completed = subprocess.run(
                self._command,
                input=serialized,
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"external command timed out after {self._timeout_seconds:g}s"
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr[:_MAX_ERROR_BODY_BYTES]
            raise RuntimeError(
                f"external command exited with {completed.returncode}: {stderr}"
            )
        if len(completed.stdout.encode("utf-8")) > self._max_stdout_bytes:
            raise ValueError("external command stdout exceeds max_stdout_bytes")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("external command stdout must be one JSON object") from exc
        if not isinstance(payload, dict):
            raise ValueError("external command stdout must be one JSON object")

        prediction = Prediction.model_validate(payload)
        metadata = {
            **prediction.metadata,
            "transport": "json-command-v1",
        }
        return prediction.model_copy(update={"metadata": metadata})


def render_model_state(item: BenchmarkItem) -> Any:
    """Return only fields a model is allowed to observe.

    Evidence relation is a benchmark annotation and is intentionally never exposed.
    """
    if not item.evidence:
        return item.state

    visible_evidence: list[dict[str, Any]] = []
    for evidence in item.evidence:
        value: dict[str, Any] = {"id": evidence.id}
        if evidence.text is not None:
            value["text"] = evidence.text
        if evidence.structured is not None:
            value["structured"] = evidence.structured
        if evidence.source_ref is not None:
            value["source_ref"] = evidence.source_ref
        visible_evidence.append(value)
    return {"state": item.state, "evidence": visible_evidence}


def _coerce_and_normalize_probabilities(
    raw: dict[Any, Any],
    *,
    expected_ids: list[str],
) -> dict[str, float]:
    if set(raw) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(raw))
        extra = sorted(set(raw) - set(expected_ids))
        raise ValueError(
            f"System-One probability keys mismatch: missing={missing}, extra={extra}"
        )

    probabilities: dict[str, float] = {}
    for action_id in expected_ids:
        value = raw[action_id]
        if isinstance(value, bool):
            raise ValueError("System-One probabilities must be numeric, not boolean")
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"System-One probability for {action_id!r} is not numeric"
            ) from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"System-One probability for {action_id!r} must be in [0, 1]"
            )
        probabilities[action_id] = probability

    total = sum(probabilities.values())
    if total <= 0.0:
        raise ValueError("System-One probabilities must have positive total mass")
    rounding_tolerance = len(expected_ids) * 5e-5 + 1e-9
    if abs(total - 1.0) > max(1e-6, rounding_tolerance):
        raise ValueError(
            "System-One probabilities are not normalized within declared "
            f"transport-rounding tolerance: sum={total:.12g}"
        )
    if abs(total - 1.0) <= 1e-6:
        return probabilities
    return {
        action_id: probability / total
        for action_id, probability in probabilities.items()
    }
