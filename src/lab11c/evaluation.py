"""User-level ranking evaluation with explicit eligibility denominators."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np
import pandas as pd

from .recommenders import RecommendationSystem
from .teaching import cosine


def precision_at_k(recommended: list[str], positives: set[str], k: int) -> float:
    return len(set(recommended[:k]) & positives) / k


def recall_at_k(recommended: list[str], positives: set[str], k: int) -> float:
    if not positives:
        raise ValueError("recall is undefined without at least one positive")
    return len(set(recommended[:k]) & positives) / len(positives)


def ndcg_at_k(recommended: list[str], positives: set[str], k: int) -> float:
    if not positives:
        raise ValueError("NDCG is undefined without at least one positive")
    gains = [1.0 if item in positives else 0.0 for item in recommended[:k]]
    dcg = sum(gain / log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


@dataclass
class RankingEvaluation:
    aggregate: dict
    user_metrics: pd.DataFrame
    recommendations: pd.DataFrame


def evaluate_ranking(
    system: RecommendationSystem,
    heldout: pd.DataFrame,
    method: str,
    alpha: float = 0.75,
    k: int = 5,
) -> RankingEvaluation:
    known_items = set(system.items)
    positives_frame = heldout[heldout["ProductID"].isin(known_items) & heldout["Rating"].ge(4)]
    positive_by_user = {
        str(user): set(group["ProductID"].astype(str))
        for user, group in positives_frame.groupby("UserID", sort=True)
    }
    user_rows = []
    recommendation_rows = []
    for user in sorted(positive_by_user):
        positives = positive_by_user[user]
        ranking = system.recommend(user, method, alpha=alpha, k=k)
        recommended = ranking["ProductID"].astype(str).tolist()
        detail = ranking.attrs.get("detail", {})
        user_rows.append(
            {
                "Method": method,
                "Alpha": alpha if method == "hybrid" else np.nan,
                "UserID": user,
                "Segment": "warm" if user in system.user_to_index else "cold",
                "EligiblePositiveCount": len(positives),
                "RecommendationCount": len(recommended),
                "Hits": len(set(recommended) & positives),
                "PrecisionAt5": precision_at_k(recommended, positives, k),
                "RecallAt5": recall_at_k(recommended, positives, k),
                "NDCGAt5": ndcg_at_k(recommended, positives, k),
                "FallbackUsed": bool(detail.get("fallback_used", False)),
                "FallbackReason": detail.get("fallback_reason"),
            }
        )
        for row in ranking.itertuples(index=False):
            recommendation_rows.append(
                {
                    "Method": method,
                    "Alpha": alpha if method == "hybrid" else np.nan,
                    "UserID": user,
                    "Segment": "warm" if user in system.user_to_index else "cold",
                    "Rank": int(row.Rank),
                    "ProductID": str(row.ProductID),
                    "Score": float(row.Score),
                    "TrainingPopularity": int(row.TrainingPopularity),
                    "IsHeldoutPositive": str(row.ProductID) in positives,
                    "FallbackUsed": bool(detail.get("fallback_used", False)),
                    "FallbackReason": detail.get("fallback_reason"),
                }
            )

    users = pd.DataFrame(user_rows)
    recommendations = pd.DataFrame(recommendation_rows)
    if users.empty:
        raise ValueError("no ranking-eligible users in heldout data")
    unique_recommended = recommendations["ProductID"].nunique()

    diversities = []
    short_lists = 0
    for _, rows in recommendations.groupby("UserID"):
        item_ids = rows.sort_values("Rank")["ProductID"].tolist()
        if len(item_ids) < 2:
            short_lists += 1
            continue
        distances = []
        for left in range(len(item_ids)):
            for right in range(left + 1, len(item_ids)):
                a = system.item_vectors[system.item_to_index[item_ids[left]]]
                b = system.item_vectors[system.item_to_index[item_ids[right]]]
                distances.append(1.0 - cosine(a, b))
        diversities.append(float(np.mean(distances)))

    def segment_summary(segment: str) -> dict:
        subset = users[users["Segment"] == segment]
        return {
            "users": int(len(subset)),
            "precision_at_5": float(subset["PrecisionAt5"].mean()) if len(subset) else None,
            "recall_at_5": float(subset["RecallAt5"].mean()) if len(subset) else None,
            "ndcg_at_5": float(subset["NDCGAt5"].mean()) if len(subset) else None,
        }

    aggregate = {
        "method": method,
        "alpha": alpha if method == "hybrid" else None,
        "eligible_users": int(len(users)),
        "eligible_positive_events": int(users["EligiblePositiveCount"].sum()),
        "precision_at_5": float(users["PrecisionAt5"].mean()),
        "recall_at_5": float(users["RecallAt5"].mean()),
        "ndcg_at_5": float(users["NDCGAt5"].mean()),
        "catalog_coverage": float(unique_recommended / len(system.items)),
        "distinct_recommended_items": int(unique_recommended),
        "proxy_catalog_size": int(len(system.items)),
        "mean_intra_list_diversity": float(np.mean(diversities)) if diversities else None,
        "diversity_eligible_lists": int(len(diversities)),
        "lists_shorter_than_two": int(short_lists),
        "fallback_users": int(users["FallbackUsed"].sum()),
        "warm": segment_summary("warm"),
        "cold": segment_summary("cold"),
    }
    return RankingEvaluation(aggregate, users, recommendations)
