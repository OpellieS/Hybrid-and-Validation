"""User-cluster bootstrap utilities for ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = ("PrecisionAt5", "RecallAt5", "NDCGAt5")


def bootstrap_model_metrics(
    user_metrics: pd.DataFrame,
    replicates: int = 10_000,
    seed: int = 202406,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    replicate_rows = []
    summary_rows = []
    for method, group in user_metrics.groupby("Method", sort=True):
        group = group.sort_values("UserID").reset_index(drop=True)
        n_users = len(group)
        draws = rng.integers(0, n_users, size=(replicates, n_users))
        for metric in METRIC_COLUMNS:
            values = group[metric].to_numpy(float)
            statistics = values[draws].mean(axis=1)
            replicate_rows.extend(
                {"Replicate": index, "Method": method, "Metric": metric, "Value": value}
                for index, value in enumerate(statistics)
            )
            low, high = np.percentile(statistics, [2.5, 97.5])
            summary_rows.append(
                {
                    "Type": "model",
                    "MethodA": method,
                    "MethodB": None,
                    "Metric": metric,
                    "EligibleUsers": n_users,
                    "PointEstimate": float(values.mean()),
                    "CI2.5": float(low),
                    "CI97.5": float(high),
                    "Replicates": replicates,
                    "Seed": seed,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(replicate_rows)


def paired_bootstrap_difference(
    user_metrics: pd.DataFrame,
    method_a: str,
    method_b: str,
    replicates: int = 10_000,
    seed: int = 202406,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = user_metrics[user_metrics["Method"] == method_a].set_index("UserID")
    right = user_metrics[user_metrics["Method"] == method_b].set_index("UserID")
    users = sorted(set(left.index) & set(right.index))
    if not users:
        raise ValueError("paired comparison has no common users")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(replicates, len(users)))
    summaries = []
    rows = []
    for metric in METRIC_COLUMNS:
        differences = left.loc[users, metric].to_numpy(float) - right.loc[users, metric].to_numpy(float)
        statistics = differences[draws].mean(axis=1)
        rows.extend(
            {"Replicate": index, "MethodA": method_a, "MethodB": method_b, "Metric": metric, "Difference": value}
            for index, value in enumerate(statistics)
        )
        low, high = np.percentile(statistics, [2.5, 97.5])
        summaries.append(
            {
                "Type": "paired_difference",
                "MethodA": method_a,
                "MethodB": method_b,
                "Metric": metric,
                "EligibleUsers": len(users),
                "PointEstimate": float(differences.mean()),
                "CI2.5": float(low),
                "CI97.5": float(high),
                "Replicates": replicates,
                "Seed": seed,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(rows)
