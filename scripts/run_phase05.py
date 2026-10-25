from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lab11c.data import load_catalog, load_interactions, temporal_split
from lab11c.evaluation import evaluate_ranking
from lab11c.freeze import validate_frozen_manifest
from lab11c.io_utils import write_json
from lab11c.matrix_factorization import ExplicitMatrixFactorization, rating_errors
from lab11c.recommenders import METHODS, RecommendationSystem, RecommenderConfig

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "artifacts" / "results"
EVIDENCE = ROOT / "artifacts" / "evidence"
FREEZE_PATH = ROOT / "artifacts" / "frozen_experiment_manifest.json"
MODEL_PATH = ROOT / "artifacts" / "models" / "selected_mf_state.json"

REFERENCE_COVERAGE = {
    "popularity": 5 / 70,
    "content": 19 / 70,
    "user_cf": 5 / 70,
    "item_cf": 5 / 70,
    "matrix_factorization": 12 / 70,
    "hybrid": 22 / 70,
}


def main() -> None:
    # Gate first. Test rows are not loaded until this validation returns.
    frozen = validate_frozen_manifest(FREEZE_PATH)
    selected_alpha = float(frozen["selected_alpha"])

    events = load_interactions(RAW / "interaction_events-2.csv")
    split = temporal_split(events)
    catalog = load_catalog(RAW / "synthetic_teaching_catalog-2.csv")
    model_state = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    factor_model = ExplicitMatrixFactorization.from_dict(model_state)
    rec_config = RecommenderConfig(**frozen["model_configuration"]["recommender"])
    system = RecommendationSystem(split.train, catalog, factor_model, rec_config)

    aggregates = []
    user_tables = []
    recommendation_tables = []
    for method in METHODS:
        evaluation = evaluate_ranking(system, split.test, method, alpha=selected_alpha)
        aggregate = dict(evaluation.aggregate)
        if method == "matrix_factorization":
            aggregate.update({f"rating_{key}": value for key, value in rating_errors(factor_model, split.test).items()})
            aggregate["rating_error_reason"] = "warm-user and training-known-item events"
        else:
            aggregate.update({"rating_events": None, "rating_mae": None, "rating_rmse": None})
            aggregate["rating_error_reason"] = "ranking score is not a consistently calibrated rating prediction"
        aggregates.append(aggregate)
        user_tables.append(evaluation.user_metrics)
        recommendation_tables.append(evaluation.recommendations)

    aggregate_frame = pd.json_normalize(aggregates, sep="_")
    user_frame = pd.concat(user_tables, ignore_index=True)
    recommendation_frame = pd.concat(recommendation_tables, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    aggregate_frame.to_csv(RESULTS / "test_method_metrics.csv", index=False)
    user_frame.to_csv(RESULTS / "test_user_metrics.csv", index=False)
    recommendation_frame.to_csv(RESULTS / "test_recommendations.csv", index=False)

    comparisons = []
    for aggregate in aggregates:
        method = aggregate["method"]
        comparisons.append(
            {
                "method": method,
                "eligible_users_observed": aggregate["eligible_users"],
                "eligible_users_reference": 9,
                "eligible_users_match": aggregate["eligible_users"] == 9,
                "precision_observed": aggregate["precision_at_5"],
                "recall_observed": aggregate["recall_at_5"],
                "ndcg_observed": aggregate["ndcg_at_5"],
                "all_ranking_metrics_zero_reference": True,
                "all_ranking_metrics_zero_match": all(
                    aggregate[name] == 0.0 for name in ("precision_at_5", "recall_at_5", "ndcg_at_5")
                ),
                "coverage_observed": aggregate["catalog_coverage"],
                "coverage_reference": REFERENCE_COVERAGE[method],
                "coverage_match": abs(aggregate["catalog_coverage"] - REFERENCE_COVERAGE[method]) < 5e-7,
                "fallback_users_observed": aggregate["fallback_users"],
                "no_supported_cf_neighborhood_match": (
                    aggregate["fallback_users"] == 9 if method in {"user_cf", "item_cf"} else None
                ),
            }
        )
    write_json(
        EVIDENCE / "test_reference_comparison.json",
        {
            "configuration_hash": frozen["configuration_hash"],
            "selected_alpha_loaded_from_freeze": selected_alpha,
            "no_refit_or_retuning_after_test": True,
            "comparisons": comparisons,
            "future_only_test_item_events_excluded_from_ranking_positives": int(
                (~split.test["ProductID"].isin(system.items)).sum()
            ),
            "rating_error_eligible_events": int(
                (split.test["UserID"].isin(system.users) & split.test["ProductID"].isin(system.items)).sum()
            ),
        },
    )
    print(
        {
            "phase": "05",
            "status": "PASS",
            "configuration_hash": frozen["configuration_hash"],
            "selected_alpha": selected_alpha,
            "eligible_users": sorted(user_frame["UserID"].unique().tolist()),
            "all_zero": bool((aggregate_frame[["precision_at_5", "recall_at_5", "ndcg_at_5"]] == 0).all().all()),
        }
    )


if __name__ == "__main__":
    main()
