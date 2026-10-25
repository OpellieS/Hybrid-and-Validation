"""Six deterministic recommenders sharing one candidate and ranking policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .matrix_factorization import ExplicitMatrixFactorization
from .teaching import centered_overlap_similarity, cosine, manual_tfidf, rating_weighted_profile

METHODS = (
    "popularity",
    "content",
    "user_cf",
    "item_cf",
    "matrix_factorization",
    "hybrid",
)


@dataclass(frozen=True)
class RecommenderConfig:
    k: int = 5
    min_overlap: int = 2
    similarity_shrinkage: float = 10.0
    max_neighbors: int = 20
    min_neighbor_support: int = 1
    positive_similarity_only: bool = True


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class RecommendationSystem:
    def __init__(
        self,
        train: pd.DataFrame,
        catalog: pd.DataFrame,
        factor_model: ExplicitMatrixFactorization,
        config: RecommenderConfig | None = None,
    ):
        self.config = config or RecommenderConfig()
        self.train = train.copy()
        self.factor_model = factor_model
        self.fit_max_time = train["EventTime"].max().isoformat()
        self.items = sorted(train["ProductID"].astype(str).unique().tolist())
        self.users = sorted(train["UserID"].astype(str).unique().tolist())
        if set(catalog["ProductID"].astype(str)) != set(self.items):
            raise ValueError("catalog must exactly equal the training-known item universe")
        self.catalog = catalog.set_index("ProductID").loc[self.items].reset_index()
        self.item_to_index = {item: index for index, item in enumerate(self.items)}
        self.user_to_index = {user: index for index, user in enumerate(self.users)}

        counts = train.groupby("ProductID").size()
        self.popularity = {item: int(counts.get(item, 0)) for item in self.items}
        self.seen = train.groupby("UserID")["ProductID"].agg(lambda values: set(map(str, values))).to_dict()

        descriptions = self.catalog["Description"].astype(str).tolist()
        self.vocabulary, self.item_vectors, self.idf = manual_tfidf(descriptions)
        self.user_profiles: dict[str, np.ndarray] = {}
        for user, history in train.groupby("UserID", sort=True):
            indices = [self.item_to_index[item] for item in history["ProductID"]]
            self.user_profiles[str(user)] = rating_weighted_profile(
                self.item_vectors[indices], history["Rating"].to_numpy(float)
            )

        self.rating_matrix = np.full((len(self.users), len(self.items)), np.nan, dtype=float)
        for row in train.itertuples(index=False):
            self.rating_matrix[self.user_to_index[str(row.UserID)], self.item_to_index[str(row.ProductID)]] = float(row.Rating)
        self.user_means = np.nanmean(self.rating_matrix, axis=1)
        self.item_means = np.nanmean(self.rating_matrix, axis=0)

    def candidates(self, user_id: str) -> list[str]:
        prior = self.seen.get(str(user_id), set())
        return [item for item in self.items if item not in prior]

    def _popularity_scores(self, candidates: list[str]) -> np.ndarray:
        return np.array([self.popularity[item] for item in candidates], dtype=float)

    def _content_scores(self, user_id: str, candidates: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        profile = self.user_profiles.get(str(user_id))
        if profile is None or float(np.linalg.norm(profile)) == 0.0:
            return self._popularity_scores(candidates), {
                "fallback_used": True,
                "fallback_reason": "unknown_user_or_zero_content_profile",
            }
        indices = [self.item_to_index[item] for item in candidates]
        scores = np.array([cosine(profile, self.item_vectors[index]) for index in indices], dtype=float)
        return scores, {"fallback_used": False, "fallback_reason": None}

    def _user_neighbors(self, user_id: str) -> list[tuple[int, float, int]]:
        target_index = self.user_to_index.get(str(user_id))
        if target_index is None:
            return []
        target = self.rating_matrix[target_index]
        neighbors = []
        for other_index, other in enumerate(self.rating_matrix):
            if other_index == target_index:
                continue
            similarity, overlap = centered_overlap_similarity(
                target, other, self.user_means[target_index], self.user_means[other_index]
            )
            if overlap < self.config.min_overlap:
                continue
            similarity *= overlap / (overlap + self.config.similarity_shrinkage)
            if self.config.positive_similarity_only and similarity <= 0.0:
                continue
            if similarity != 0.0:
                neighbors.append((other_index, float(similarity), overlap))
        neighbors.sort(key=lambda value: (-value[1], self.users[value[0]]))
        return neighbors[: self.config.max_neighbors]

    def _user_cf_scores(self, user_id: str, candidates: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        target_index = self.user_to_index.get(str(user_id))
        neighbors = self._user_neighbors(user_id)
        if target_index is None or not neighbors:
            return self._popularity_scores(candidates), {
                "fallback_used": True,
                "fallback_reason": "unknown_user_or_no_supported_user_neighbors",
                "supported_neighbors": 0,
            }
        scores = np.full(len(candidates), float(self.user_means[target_index]), dtype=float)
        support = np.zeros(len(candidates), dtype=int)
        for position, item in enumerate(candidates):
            item_index = self.item_to_index[item]
            contributions = [
                (similarity, self.rating_matrix[neighbor_index, item_index] - self.user_means[neighbor_index])
                for neighbor_index, similarity, _ in neighbors
                if not np.isnan(self.rating_matrix[neighbor_index, item_index])
            ]
            support[position] = len(contributions)
            if len(contributions) >= self.config.min_neighbor_support:
                denominator = sum(abs(similarity) for similarity, _ in contributions)
                scores[position] += sum(similarity * deviation for similarity, deviation in contributions) / denominator
        return scores, {
            "fallback_used": False,
            "fallback_reason": None,
            "supported_neighbors": len(neighbors),
            "candidate_support": support,
        }

    def _item_similarity(self, left_index: int, right_index: int) -> tuple[float, int]:
        similarity, overlap = centered_overlap_similarity(
            self.rating_matrix[:, left_index],
            self.rating_matrix[:, right_index],
            self.item_means[left_index],
            self.item_means[right_index],
        )
        if overlap < self.config.min_overlap:
            return 0.0, overlap
        similarity *= overlap / (overlap + self.config.similarity_shrinkage)
        if self.config.positive_similarity_only and similarity <= 0.0:
            return 0.0, overlap
        return float(similarity), overlap

    def _item_cf_scores(self, user_id: str, candidates: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        target_index = self.user_to_index.get(str(user_id))
        if target_index is None:
            return self._popularity_scores(candidates), {
                "fallback_used": True,
                "fallback_reason": "unknown_user",
                "supported_candidates": 0,
            }
        rated_indices = np.flatnonzero(~np.isnan(self.rating_matrix[target_index]))
        scores = np.full(len(candidates), float(self.user_means[target_index]), dtype=float)
        support = np.zeros(len(candidates), dtype=int)
        for position, item in enumerate(candidates):
            candidate_index = self.item_to_index[item]
            neighbors = []
            for rated_index in rated_indices:
                similarity, _ = self._item_similarity(candidate_index, int(rated_index))
                if similarity != 0.0:
                    neighbors.append((int(rated_index), similarity))
            neighbors.sort(key=lambda value: (-value[1], self.items[value[0]]))
            neighbors = neighbors[: self.config.max_neighbors]
            support[position] = len(neighbors)
            if len(neighbors) >= self.config.min_neighbor_support:
                denominator = sum(abs(similarity) for _, similarity in neighbors)
                scores[position] = self.item_means[candidate_index] + sum(
                    similarity * (self.rating_matrix[target_index, rated_index] - self.item_means[rated_index])
                    for rated_index, similarity in neighbors
                ) / denominator
        if int((support >= self.config.min_neighbor_support).sum()) == 0:
            return self._popularity_scores(candidates), {
                "fallback_used": True,
                "fallback_reason": "no_supported_item_neighborhoods",
                "supported_candidates": 0,
                "candidate_support": support,
            }
        return scores, {
            "fallback_used": False,
            "fallback_reason": None,
            "supported_candidates": int((support >= self.config.min_neighbor_support).sum()),
            "candidate_support": support,
        }

    def _factor_scores(self, user_id: str, candidates: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        if str(user_id) not in self.factor_model.user_to_index:
            return self._popularity_scores(candidates), {
                "fallback_used": True,
                "fallback_reason": "unknown_user",
            }
        return np.array(
            [self.factor_model.predict_one(str(user_id), item, clip=False) for item in candidates], dtype=float
        ), {"fallback_used": False, "fallback_reason": None}

    def score_candidates(self, user_id: str, method: str, alpha: float = 0.75) -> pd.DataFrame:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        user_id = str(user_id)
        candidates = self.candidates(user_id)
        popularity = self._popularity_scores(candidates)
        detail: dict[str, Any] = {}
        content_raw = np.full(len(candidates), np.nan)
        factor_raw = np.full(len(candidates), np.nan)
        content_scaled = np.full(len(candidates), np.nan)
        factor_scaled = np.full(len(candidates), np.nan)

        if method == "popularity":
            score = popularity.copy()
            detail = {"fallback_used": False, "fallback_reason": None}
        elif method == "content":
            score, detail = self._content_scores(user_id, candidates)
        elif method == "user_cf":
            score, detail = self._user_cf_scores(user_id, candidates)
        elif method == "item_cf":
            score, detail = self._item_cf_scores(user_id, candidates)
        elif method == "matrix_factorization":
            score, detail = self._factor_scores(user_id, candidates)
        else:
            if user_id not in self.user_to_index:
                score = popularity.copy()
                detail = {"fallback_used": True, "fallback_reason": "unknown_user"}
            else:
                content_raw, content_detail = self._content_scores(user_id, candidates)
                factor_raw, factor_detail = self._factor_scores(user_id, candidates)
                content_scaled = minmax_scale(content_raw)
                factor_scaled = minmax_scale(factor_raw)
                score = alpha * content_scaled + (1.0 - alpha) * factor_scaled
                detail = {
                    "fallback_used": False,
                    "fallback_reason": None,
                    "content_detail": content_detail,
                    "factor_detail": factor_detail,
                }

        trace = pd.DataFrame(
            {
                "UserID": user_id,
                "ProductID": candidates,
                "Method": method,
                "Score": score,
                "TrainingPopularity": popularity.astype(int),
                "ContentRaw": content_raw,
                "FactorRaw": factor_raw,
                "ContentScaled": content_scaled,
                "FactorScaled": factor_scaled,
                "Alpha": alpha if method == "hybrid" else np.nan,
            }
        )
        trace.attrs["detail"] = detail
        return trace

    def recommend(self, user_id: str, method: str, alpha: float = 0.75, k: int | None = None) -> pd.DataFrame:
        limit = self.config.k if k is None else int(k)
        if limit < 0:
            raise ValueError("k cannot be negative")
        trace = self.score_candidates(user_id, method, alpha)
        ranked = trace.sort_values(
            ["Score", "TrainingPopularity", "ProductID"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(limit).copy()
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        if ranked["ProductID"].duplicated().any():
            raise AssertionError("ranking contains duplicate ProductIDs")
        if set(ranked["ProductID"]) & self.seen.get(str(user_id), set()):
            raise AssertionError("ranking contains training-seen items")
        ranked.attrs["detail"] = trace.attrs["detail"]
        return ranked.reset_index(drop=True)

    def fitted_summary(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "training_max_time": self.fit_max_time,
            "candidate_count": len(self.items),
            "user_count": len(self.users),
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "popularity": self.popularity,
            "profile_users": sorted(self.user_profiles),
        }
