"""Temporal leakage assertions for fitted objects and selection decisions."""

from __future__ import annotations

import pandas as pd


def assert_strictly_before(name: str, latest_time: str | pd.Timestamp, boundary: str | pd.Timestamp) -> dict:
    latest = pd.Timestamp(latest_time)
    cutoff = pd.Timestamp(boundary)
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    if not latest < cutoff:
        raise AssertionError(f"{name} uses {latest.isoformat()}, not strictly before {cutoff.isoformat()}")
    return {
        "object": name,
        "latest_information_time": latest.isoformat(),
        "required_boundary_exclusive": cutoff.isoformat(),
        "pass": True,
    }


def assert_candidate_universe(training_items: set[str], candidate_items: set[str], future_only_items: set[str]) -> dict:
    if candidate_items != training_items:
        raise AssertionError("candidate universe differs from the training-known item set")
    leaked = sorted(candidate_items & future_only_items)
    if leaked:
        raise AssertionError(f"candidate universe contains future-only items: {leaked}")
    return {
        "object": "candidate_universe",
        "training_item_count": len(training_items),
        "candidate_item_count": len(candidate_items),
        "future_only_overlap": leaked,
        "pass": True,
    }
