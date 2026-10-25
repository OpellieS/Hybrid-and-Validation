"""Transparent numerical building blocks used by tests and the notebook."""

from __future__ import annotations

from collections import Counter
from math import log

import numpy as np


def observed_mask(values: np.ndarray) -> np.ndarray:
    return ~np.isnan(np.asarray(values, dtype=float))


def observed_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.nanmean(np.asarray(values, dtype=float), axis=axis)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def logarithmic_tf(count: int) -> float:
    if count < 0:
        raise ValueError("term count cannot be negative")
    return 0.0 if count == 0 else 1.0 + log(count)


def inverse_document_frequency(document_count: int, document_frequency: int) -> float:
    if not 1 <= document_frequency <= document_count:
        raise ValueError("document frequency must be in [1, document_count]")
    return log(document_count / document_frequency)


def manual_tfidf(documents: list[str]) -> tuple[list[str], np.ndarray, dict[str, float]]:
    counts = [term_counts(document) for document in documents]
    vocabulary = sorted({term for row in counts for term in row})
    n_docs = len(documents)
    idf = {
        term: inverse_document_frequency(n_docs, sum(term in row for row in counts))
        for term in vocabulary
    }
    matrix = np.array(
        [[logarithmic_tf(row.get(term, 0)) * idf[term] for term in vocabulary] for row in counts],
        dtype=float,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return vocabulary, matrix, idf


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def rating_weighted_profile(item_vectors: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(item_vectors, dtype=float)
    weights = np.asarray(ratings, dtype=float)
    if vectors.shape[0] != weights.shape[0]:
        raise ValueError("one rating is required per item vector")
    denominator = float(np.abs(weights).sum())
    if denominator == 0.0:
        return np.zeros(vectors.shape[1], dtype=float)
    profile = np.average(vectors, axis=0, weights=np.abs(weights))
    norm = float(np.linalg.norm(profile))
    return profile if norm == 0.0 else profile / norm


def centered_overlap_similarity(
    left: np.ndarray,
    right: np.ndarray,
    left_mean: float,
    right_mean: float,
) -> tuple[float, int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    overlap = ~np.isnan(left) & ~np.isnan(right)
    n_overlap = int(overlap.sum())
    if n_overlap == 0:
        return 0.0, 0
    return cosine(left[overlap] - left_mean, right[overlap] - right_mean), n_overlap
