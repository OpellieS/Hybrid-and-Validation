from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lab11c.data import file_sha256, load_catalog, load_interactions, temporal_split
from lab11c.evaluation import evaluate_ranking
from lab11c.freeze import freeze_manifest
from lab11c.io_utils import object_sha256, write_json
from lab11c.matrix_factorization import ExplicitMatrixFactorization, rating_errors
from lab11c.recommenders import METHODS, RecommendationSystem, RecommenderConfig

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
EVIDENCE = ROOT / "artifacts" / "evidence"
RESULTS = ROOT / "artifacts" / "results"
MODEL_PATH = ROOT / "artifacts" / "models" / "selected_mf_state.json"
FREEZE_PATH = ROOT / "artifacts" / "frozen_experiment_manifest.json"

ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


def main() -> None:
    events = load_interactions(RAW / "interaction_events-2.csv")
    split = temporal_split(events)
    catalog = load_catalog(RAW / "synthetic_teaching_catalog-2.csv")
    model_state = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    factor_model = ExplicitMatrixFactorization.from_dict(model_state)
    rec_config = RecommenderConfig()
    system = RecommendationSystem(split.train, catalog, factor_model, rec_config)

    alpha_rows = []
    alpha_evaluations = {}
    for alpha in ALPHA_GRID:
        evaluation = evaluate_ranking(system, split.validation, "hybrid", alpha=alpha)
        alpha_evaluations[alpha] = evaluation
        alpha_rows.append(evaluation.aggregate)
    alpha_record = pd.DataFrame(alpha_rows)
    alpha_record["distance_from_balanced"] = (alpha_record["alpha"] - 0.5).abs()
    alpha_record = alpha_record.sort_values(
        ["ndcg_at_5", "recall_at_5", "precision_at_5", "distance_from_balanced", "alpha"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    alpha_record["selected"] = False
    alpha_record.loc[0, "selected"] = True
    selected_alpha = float(alpha_record.loc[0, "alpha"])

    method_aggregates = []
    user_tables = []
    recommendation_tables = []
    for method in METHODS:
        evaluation = (
            alpha_evaluations[selected_alpha]
            if method == "hybrid"
            else evaluate_ranking(system, split.validation, method, alpha=selected_alpha)
        )
        method_aggregates.append(evaluation.aggregate)
        user_tables.append(evaluation.user_metrics)
        recommendation_tables.append(evaluation.recommendations)

    RESULTS.mkdir(parents=True, exist_ok=True)
    alpha_record.to_csv(RESULTS / "validation_alpha_search.csv", index=False)
    pd.json_normalize(method_aggregates, sep="_").to_csv(RESULTS / "validation_method_metrics.csv", index=False)
    pd.concat(user_tables, ignore_index=True).to_csv(RESULTS / "validation_user_metrics.csv", index=False)
    pd.concat(recommendation_tables, ignore_index=True).to_csv(
        RESULTS / "validation_recommendations.csv", index=False
    )
    write_json(
        EVIDENCE / "validation_selection.json",
        {
            "alpha_grid_predeclared": ALPHA_GRID,
            "selection_order": [
                "NDCG@5 descending",
                "Recall@5 descending",
                "Precision@5 descending",
                "distance from balanced alpha=0.5 ascending",
                "alpha ascending",
            ],
            "selected_alpha": selected_alpha,
            "reference_alpha": 0.75,
            "reference_alpha_forced": False,
            "ranking_eligible_users": int(alpha_record.loc[0, "eligible_users"]),
            "mf_rating_errors": rating_errors(factor_model, split.validation),
            "test_performance_evaluated": False,
        },
    )

    packages = ["numpy", "pandas", "scipy", "scikit-learn"]
    fitted_payloads = {
        "matrix_factor_state": model_state,
        "popularity": system.popularity,
        "content": {
            "vocabulary": system.vocabulary,
            "idf": system.idf,
            "item_vectors": system.item_vectors,
            "profiles": system.user_profiles,
        },
        "rating_matrix": {
            "observed_mask": np.isfinite(system.rating_matrix).astype(int),
            "observed_values": np.nan_to_num(system.rating_matrix, nan=0.0),
        },
        "candidate_items": system.items,
    }
    manifest_payload = {
        "status": "CONFIG_FROZEN",
        "created_before_test_evaluation": True,
        "training_boundary_exclusive": "2024-05-21T00:00:00+00:00",
        "validation_boundary_exclusive": "2024-06-13T00:00:00+00:00",
        "fitted_max_event_time": system.fit_max_time,
        "model_configuration": {
            "matrix_factorization": model_state["config"],
            "recommender": rec_config.__dict__,
            "hybrid_formula": "alpha * scaled_content + (1-alpha) * scaled_factor",
            "alpha_grid": ALPHA_GRID,
        },
        "selected_alpha": selected_alpha,
        "seeds": {"matrix_factorization": model_state["config"]["seed"]},
        "candidate_policy": "exactly 70 training-known products minus the request user's training-seen products; future-only items excluded",
        "fallback_policy": "unknown users and zero/unsupported request models route to descending training popularity",
        "metric_definitions": {
            "positive": "heldout Rating >= 4 and ProductID training-known",
            "precision_at_5": "hits divided by fixed K=5, macro mean over eligible users",
            "recall_at_5": "hits divided by each user's eligible positives, macro mean",
            "ndcg_at_5": "binary DCG with log2(rank+1) discount divided by ideal DCG, macro mean",
            "rating_error": "MAE/RMSE on warm-user and training-known-item events; predictions clipped to [1,5]",
            "coverage": "distinct recommended items divided by 70 training-known products",
            "diversity": "mean pairwise 1-content-cosine within each distinct list, then mean across eligible lists",
        },
        "tie_breaking": [
            "recommendation score descending",
            "training popularity descending",
            "ProductID lexical ascending",
        ],
        "dataset_hashes": {path.name: file_sha256(path) for path in sorted(RAW.glob("*.csv"))},
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **{package: importlib.metadata.version(package) for package in packages},
        },
        "fitted_state_checksums": {
            name: object_sha256(value) for name, value in fitted_payloads.items()
        },
        "validation_artifacts": {
            "alpha_search_sha256": file_sha256(RESULTS / "validation_alpha_search.csv"),
            "method_metrics_sha256": file_sha256(RESULTS / "validation_method_metrics.csv"),
        },
        "known_reference_not_used_for_selection": {
            "reported_alpha": 0.75,
            "reported_test_metrics": "all ranking metrics zero",
            "reported_coverages": [0.071429, 0.271429, 0.171429, 0.314286],
        },
    }
    manifest = freeze_manifest(FREEZE_PATH, manifest_payload)
    (EVIDENCE / "CONFIG_FROZEN").write_text(manifest["configuration_hash"] + "\n", encoding="ascii")
    print(
        {
            "phase": "04",
            "status": "PASS",
            "gate": "CONFIG_FROZEN",
            "selected_alpha": selected_alpha,
            "configuration_hash": manifest["configuration_hash"],
        }
    )


if __name__ == "__main__":
    main()
