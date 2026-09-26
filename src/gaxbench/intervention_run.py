from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gaxbench.ecal import ExperimentContext
from gaxbench.interventions import InterventionManifest, intervention_manifest_sha256
from gaxbench.provenance import canonical_json_sha256
from gaxbench.schema import BenchmarkItem, Prediction


@dataclass(frozen=True)
class InterventionRunManifest:
    schema_version: str
    git_sha: str
    compute_provenance: str
    items_sha256: str
    predictions_sha256: str
    intervention_manifest_sha256: str
    stability_tv_threshold: float
    evaluator_revision: str


def build_intervention_run_manifest(
    items: list[BenchmarkItem],
    predictions: list[Prediction],
    manifest: InterventionManifest,
    *,
    context: ExperimentContext,
    stability_tv_threshold: float,
) -> InterventionRunManifest:
    if not 0.0 <= stability_tv_threshold <= 1.0 or not math.isfinite(stability_tv_threshold):
        raise ValueError("stability_tv_threshold must be finite and in [0, 1]")
    return InterventionRunManifest(
        schema_version="0.1",
        git_sha=context.git_sha,
        compute_provenance=context.compute_provenance,
        items_sha256=_models_sha256(items, key="id"),
        predictions_sha256=_models_sha256(predictions, key="item_id"),
        intervention_manifest_sha256=intervention_manifest_sha256(manifest),
        stability_tv_threshold=stability_tv_threshold,
        evaluator_revision="gax-p06-interventions-v0.1",
    )


def _models_sha256(models: list[Any], *, key: str) -> str:
    payload = [
        model.model_dump(mode="json")
        for model in sorted(models, key=lambda model: str(getattr(model, key)))
    ]
    return canonical_json_sha256(payload)
