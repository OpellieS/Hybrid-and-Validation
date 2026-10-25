import pandas as pd
import pytest

from lab11c.audit import assert_candidate_universe, assert_strictly_before
from lab11c.bootstrap import paired_bootstrap_difference


def test_temporal_assertion_and_deliberate_contamination():
    result = assert_strictly_before("clean", "2024-05-20T18:42:23+00:00", "2024-05-21T00:00:00+00:00")
    assert result["pass"] is True
    with pytest.raises(AssertionError, match="not strictly before"):
        assert_strictly_before("contaminated_copy", "2024-06-13T00:30:36+00:00", "2024-05-21T00:00:00+00:00")


def test_candidate_universe_rejects_future_only_item():
    assert_candidate_universe({"a", "b"}, {"a", "b"}, {"future"})
    with pytest.raises(AssertionError):
        assert_candidate_universe({"a", "b"}, {"a", "future"}, {"future"})


def test_paired_bootstrap_samples_same_users():
    rows = []
    for method, values in {"A": [0.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0]}.items():
        for user, value in zip(["u1", "u2", "u3"], values):
            rows.append(
                {
                    "Method": method,
                    "UserID": user,
                    "PrecisionAt5": value,
                    "RecallAt5": value,
                    "NDCGAt5": value,
                }
            )
    summary, replicates = paired_bootstrap_difference(pd.DataFrame(rows), "A", "B", replicates=100, seed=3)
    assert set(summary["EligibleUsers"]) == {3}
    assert len(replicates) == 300
    assert set(summary["PointEstimate"]) == {1 / 3}
