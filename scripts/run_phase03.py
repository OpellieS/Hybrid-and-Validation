from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lab11c.data import load_catalog, load_interactions, temporal_split
from lab11c.io_utils import write_json
from lab11c.matrix_factorization import ExplicitMatrixFactorization
from lab11c.recommenders import METHODS, RecommendationSystem, RecommenderConfig

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
EVIDENCE = ROOT / "artifacts" / "evidence"
RESULTS = ROOT / "artifacts" / "results"


def clean_records(frame: pd.DataFrame) -> list[dict]:
    return frame.where(pd.notna(frame), None).to_dict("records")


def main() -> None:
    split = temporal_split(load_interactions(RAW / "interaction_events-2.csv"))
    catalog = load_catalog(RAW / "synthetic_teaching_catalog-2.csv")
    state = json.loads((ROOT / "artifacts" / "models" / "selected_mf_state.json").read_text(encoding="utf-8"))
    factor_model = ExplicitMatrixFactorization.from_dict(state)
    config = RecommenderConfig()
    recommender = RecommendationSystem(split.train, catalog, factor_model, config)

    # Trace one validation-eligible warm user without accessing final-test performance.
    known_items = set(recommender.items)
    warm_users = set(recommender.users)
    eligible = split.validation[
        split.validation["UserID"].isin(warm_users)
        & split.validation["ProductID"].isin(known_items)
        & split.validation["Rating"].ge(4)
    ]
    trace_user = sorted(eligible["UserID"].unique().tolist())[0]
    all_scores = []
    all_rankings = []
    request_details = {}
    for method in METHODS:
        scores = recommender.score_candidates(trace_user, method, alpha=0.75)
        ranking = recommender.recommend(trace_user, method, alpha=0.75)
        all_scores.append(scores)
        all_rankings.append(ranking)
        request_details[method] = ranking.attrs["detail"]
    score_trace = pd.concat(all_scores, ignore_index=True)
    ranking_trace = pd.concat(all_rankings, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    score_trace.to_csv(RESULTS / "phase03_score_trace.csv", index=False)
    ranking_trace.to_csv(RESULTS / "phase03_ranking_trace.csv", index=False)
    write_json(
        EVIDENCE / "recommender_configuration.json",
        {
            "independent_variant": True,
            "reason": "companion script and Week 12 reference were unavailable",
            "config": config.__dict__,
            "candidate_policy": "training-known products only; remove request user's training-seen products",
            "fallback_policy": "unknown users and unsupported/zero-profile requests use training popularity",
            "tie_break": ["score descending", "training popularity descending", "ProductID lexical ascending"],
            "hybrid_scaling": "per-request min-max over eligible candidate model scores; constants map to zero",
            "trace_user": trace_user,
            "request_details": request_details,
            "fitted_summary": recommender.fitted_summary(),
        },
    )
    print(
        {
            "phase": "03",
            "status": "PASS",
            "methods": len(METHODS),
            "trace_user": trace_user,
            "score_rows": len(score_trace),
            "ranking_rows": len(ranking_trace),
        }
    )


if __name__ == "__main__":
    main()
