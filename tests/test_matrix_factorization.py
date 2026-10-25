from pathlib import Path

import numpy as np

from lab11c.data import load_interactions, temporal_split
from lab11c.matrix_factorization import ExplicitMatrixFactorization, MFConfig, rating_errors

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data" / "raw" / "interaction_events-2.csv"


def test_printed_one_event_update_uses_old_state():
    updated = ExplicitMatrixFactorization.one_event_update(
        rating=4.0,
        mu=3.0,
        user_bias=0.2,
        item_bias=-0.1,
        user_factors=np.array([0.2, 0.4]),
        item_factors=np.array([0.5, 0.1]),
        learning_rate=0.05,
        regularization=0.1,
    )
    np.testing.assert_allclose(updated["prediction"], 3.24)
    np.testing.assert_allclose(updated["residual"], 0.76)
    np.testing.assert_allclose(updated["user_bias"], 0.2370)
    np.testing.assert_allclose(updated["item_bias"], -0.0615)
    np.testing.assert_allclose(updated["user_factors"], [0.2180, 0.4018])
    np.testing.assert_allclose(updated["item_factors"], [0.5051, 0.1147])


def test_fit_objective_serialization_dimensions_and_fallbacks():
    train = temporal_split(load_interactions(SOURCE)).train
    config = MFConfig(factors=3, learning_rate=0.5, regularization=0.001, epochs=100, seed=202411)
    model = ExplicitMatrixFactorization(config).fit(train)

    assert model.history[-1]["objective"] < model.history[0]["objective"]
    assert model.mu == train["Rating"].mean()
    assert model.user_factors.shape == (train["UserID"].nunique(), 3)
    assert model.item_factors.shape == (train["ProductID"].nunique(), 3)
    assert np.isfinite(model.predict_frame(train)).all()

    state = model.to_dict()
    rebuilt = ExplicitMatrixFactorization.from_dict(state)
    row = train.iloc[0]
    np.testing.assert_allclose(
        rebuilt.predict_one(row.UserID, row.ProductID),
        model.predict_one(row.UserID, row.ProductID),
    )
    np.testing.assert_allclose(model.predict_one("UNKNOWN", "UNKNOWN"), model.mu)
    known_user = model.user_ids[0]
    known_item = model.item_ids[0]
    np.testing.assert_allclose(
        model.predict_one(known_user, "UNKNOWN"), model.mu + model.user_bias[model.user_to_index[known_user]]
    )
    np.testing.assert_allclose(
        model.predict_one("UNKNOWN", known_item), model.mu + model.item_bias[model.item_to_index[known_item]]
    )


def test_rating_errors_exclude_unknown_ids():
    split = temporal_split(load_interactions(SOURCE))
    model = ExplicitMatrixFactorization(
        MFConfig(factors=2, learning_rate=0.5, regularization=0.001, epochs=10, seed=7)
    ).fit(split.train)
    result = rating_errors(model, split.validation)
    assert result["events"] == 17
    assert np.isfinite(result["mae"])
    assert np.isfinite(result["rmse"])
