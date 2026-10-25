import json
from pathlib import Path

import numpy as np

from lab11c.data import load_catalog, load_interactions, temporal_split
from lab11c.matrix_factorization import ExplicitMatrixFactorization
from lab11c.recommenders import METHODS, RecommendationSystem, minmax_scale

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"


def system():
    split = temporal_split(load_interactions(RAW / "interaction_events-2.csv"))
    state = json.loads((ROOT / "artifacts" / "models" / "selected_mf_state.json").read_text(encoding="utf-8"))
    model = ExplicitMatrixFactorization.from_dict(state)
    return RecommendationSystem(split.train, load_catalog(RAW / "synthetic_teaching_catalog-2.csv"), model)


def test_minmax_constant_and_endpoints():
    np.testing.assert_allclose(minmax_scale(np.array([4.0, 4.0])), [0.0, 0.0])
    np.testing.assert_allclose(minmax_scale(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])


def test_all_methods_enforce_candidate_and_ranking_contract():
    rec = system()
    user = "20"
    seen = rec.seen[user]
    for method in METHODS:
        ranking = rec.recommend(user, method, alpha=0.75)
        assert 0 < len(ranking) <= 5
        assert ranking["ProductID"].is_unique
        assert not set(ranking["ProductID"]) & seen
        assert set(ranking["ProductID"]) <= set(rec.items)
        expected = ranking.sort_values(
            ["Score", "TrainingPopularity", "ProductID"], ascending=[False, False, True], kind="mergesort"
        )["ProductID"].tolist()
        assert ranking["ProductID"].tolist() == expected


def test_unknown_users_route_every_method_to_popularity():
    rec = system()
    baseline = rec.recommend("NEVER_SEEN", "popularity")["ProductID"].tolist()
    for method in METHODS:
        assert rec.recommend("NEVER_SEEN", method, alpha=0.25)["ProductID"].tolist() == baseline


def test_hybrid_endpoints_equal_scaled_components_without_labels():
    rec = system()
    user = "20"
    zero = rec.score_candidates(user, "hybrid", alpha=0.0)
    one = rec.score_candidates(user, "hybrid", alpha=1.0)
    np.testing.assert_allclose(zero["Score"], zero["FactorScaled"])
    np.testing.assert_allclose(one["Score"], one["ContentScaled"])
    assert "Rating" not in zero.columns


def test_ties_use_popularity_then_lexical_product_id():
    rec = system()
    ranking = rec.recommend("NEVER_SEEN", "popularity")
    pairs = list(zip(ranking["Score"], ranking["TrainingPopularity"], ranking["ProductID"]))
    assert pairs == sorted(pairs, key=lambda value: (-value[0], -value[1], value[2]))
