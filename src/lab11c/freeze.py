"""Immutable experiment-manifest creation and gate validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import object_sha256, write_json


def freeze_manifest(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    if "configuration_hash" in payload:
        raise ValueError("payload must not contain a precomputed configuration hash")
    manifest = dict(payload)
    manifest["configuration_hash"] = object_sha256(payload)
    write_json(path, manifest)
    validate_frozen_manifest(path)
    return manifest


def validate_frozen_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("frozen experiment manifest does not exist")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.pop("configuration_hash", None)
    if recorded is None or recorded != object_sha256(manifest):
        raise ValueError("frozen experiment manifest hash mismatch")
    manifest["configuration_hash"] = recorded
    if manifest.get("status") != "CONFIG_FROZEN":
        raise ValueError("experiment configuration is not frozen")
    required = {
        "model_configuration",
        "selected_alpha",
        "seeds",
        "candidate_policy",
        "fallback_policy",
        "metric_definitions",
        "tie_breaking",
        "dataset_hashes",
        "software_versions",
        "fitted_state_checksums",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"frozen experiment manifest missing keys: {missing}")
    return manifest
