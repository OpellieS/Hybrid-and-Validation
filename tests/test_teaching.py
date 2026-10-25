from pathlib import Path
from math import log, sqrt

import numpy as np
import pandas as pd

from lab11c.teaching import (
    centered_overlap_similarity,
    cosine,
    logarithmic_tf,
    manual_tfidf,
    observed_mask,
    observed_mean,
    rating_weighted_profile,
    term_counts,
)

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"


def test_observed_entry_mask_and_means():
    ratings = pd.read_csv(RAW / "tiny_ratings-2.csv").set_index("UserID").to_numpy(float)
    mask = observed_mask(ratings)
    assert mask.sum() == 19
    assert mask[0].tolist() == [True, True, True, False]
    assert observed_mean(ratings) == np.mean(ratings[mask])
    np.testing.assert_allclose(observed_mean(ratings, axis=1), [4.0, 3.75, 2.25, 4.0, 2.75])


def test_term_count_log_tf_idf_normalization_and_cosine():
    catalog = pd.read_csv(RAW / "tiny_content_catalog-2.csv")
    documents = catalog["Description"].tolist()
    counts = term_counts(documents[0])
    assert counts == {"coffee": 2, "roast": 1}
    assert logarithmic_tf(2) == 1.0 + log(2.0)

    vocabulary, matrix, idf = manual_tfidf(documents)
    assert vocabulary == ["coffee", "herbal", "roast", "tea"]
    assert idf["coffee"] == log(4 / 3)
    assert idf["roast"] == log(4 / 2)
    assert idf["tea"] == log(4 / 2)
    assert idf["herbal"] == log(4 / 2)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(4))
    assert cosine(matrix[0], matrix[1]) == np.dot(matrix[0], matrix[1])
    assert cosine(matrix[0], matrix[2]) == 0.0


def test_rating_weighted_profile_and_cf_teaching_calculation():
    _, matrix, _ = manual_tfidf(["coffee coffee roast", "coffee roast roast", "tea tea herbal", "coffee tea herbal"])
    profile = rating_weighted_profile(matrix[:3], np.array([5.0, 3.0, 4.0]))
    assert np.isfinite(profile).all()
    np.testing.assert_allclose(np.linalg.norm(profile), 1.0)

    ratings = pd.read_csv(RAW / "tiny_ratings-2.csv").set_index("UserID").to_numpy(float)
    means = np.nanmean(ratings, axis=1)
    similarity, overlap = centered_overlap_similarity(ratings[0], ratings[1], means[0], means[1])
    expected = 2.0 / (sqrt(2.0) * sqrt(3.1875))
    assert overlap == 3
    np.testing.assert_allclose(similarity, expected)
