import json
from pathlib import Path

import numpy as np
import pytest

from lab11c.evaluation import ndcg_at_k, precision_at_k, recall_at_k
from lab11c.freeze import freeze_manifest, validate_frozen_manifest


def test_ranking_metrics_use_fixed_k_and_binary_relevance():
    recommended = ["a", "b", "c", "d", "e"]
    positives = {"b", "d"}
    assert precision_at_k(recommended, positives, 5) == 0.4
    assert recall_at_k(recommended, positives, 5) == 1.0
    expected_dcg = 1 / np.log2(3) + 1 / np.log2(5)
    expected_idcg = 1 + 1 / np.log2(3)
    np.testing.assert_allclose(ndcg_at_k(recommended, positives, 5), expected_dcg / expected_idcg)


def test_metric_rejects_empty_positive_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 5)


def minimal_payload():
    return {
        "status": "CONFIG_FROZEN",
        "model_configuration": {},
        "selected_alpha": 0.5,
        "seeds": {},
        "candidate_policy": "train only",
        "fallback_policy": "popularity",
        "metric_definitions": {},
        "tie_breaking": [],
        "dataset_hashes": {},
        "software_versions": {},
        "fitted_state_checksums": {},
    }


def test_freeze_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "freeze.json"
    written = freeze_manifest(path, minimal_payload())
    loaded = validate_frozen_manifest(path)
    assert written == loaded
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["selected_alpha"] = 0.75
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_manifest(path)
