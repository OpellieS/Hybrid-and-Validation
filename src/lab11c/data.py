"""Strict source ingestion and the immutable temporal protocol."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("UserID", "ProductID", "Rating", "Timestamp")
TRAIN_END = pd.Timestamp("2024-05-21 00:00:00", tz="UTC")
TEST_START = pd.Timestamp("2024-06-13 00:00:00", tz="UTC")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_nonblank(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = frame[list(columns)].isna()
    blank = frame[list(columns)].apply(lambda col: col.astype("string").str.strip().eq(""))
    if bool((missing | blank).any().any()):
        bad = (missing | blank).any(axis=1)
        raise ValueError(f"missing or blank required values at rows {frame.index[bad].tolist()}")


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Read and validate the four-column interaction source without coercing IDs."""
    path = Path(path)
    raw = pd.read_csv(path, dtype="string", keep_default_na=True)
    if tuple(raw.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"expected columns {REQUIRED_COLUMNS}, got {tuple(raw.columns)}")
    _require_nonblank(raw, REQUIRED_COLUMNS)

    for column in ("UserID", "ProductID"):
        raw[column] = raw[column].str.strip()

    rating_text = raw["Rating"].str.strip()
    if not bool(rating_text.str.fullmatch(r"[+-]?\d+").all()):
        raise ValueError("Rating must contain integer text only")
    ratings = pd.to_numeric(rating_text, errors="raise", downcast="integer")
    if not bool(ratings.between(1, 5).all()):
        raise ValueError("Rating must be an integer in [1, 5]")

    timestamp_text = raw["Timestamp"].str.strip()
    if not bool(timestamp_text.str.fullmatch(r"\d+").all()):
        raise ValueError("Timestamp must contain nonnegative Unix-second integers")
    timestamps = pd.to_numeric(timestamp_text, errors="raise").astype("int64")
    parsed = pd.to_datetime(timestamps, unit="s", utc=True, errors="raise")

    duplicate = raw.duplicated(["UserID", "ProductID"], keep=False)
    if bool(duplicate.any()):
        pairs = raw.loc[duplicate, ["UserID", "ProductID"]].to_dict("records")
        raise ValueError(f"duplicate user-item pairs: {pairs}")

    frame = pd.DataFrame(
        {
            "SourceRow": range(len(raw)),
            "UserID": raw["UserID"].astype(str),
            "ProductID": raw["ProductID"].astype(str),
            "Rating": ratings.astype("int8"),
            "Timestamp": timestamps,
            "EventTime": parsed,
        }
    )
    if not frame["EventTime"].notna().all():
        raise ValueError("Timestamp conversion produced missing values")
    return frame


def load_catalog(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    expected = ("ProductID", "Description", "MetadataStatus")
    if tuple(frame.columns) != expected:
        raise ValueError(f"expected catalog columns {expected}, got {tuple(frame.columns)}")
    _require_nonblank(frame, expected)
    frame = frame.astype({column: str for column in expected})
    frame["ProductID"] = frame["ProductID"].str.strip()
    if bool(frame["ProductID"].duplicated().any()):
        raise ValueError("catalog ProductID must be unique")
    return frame


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame) -> SplitData:
    train = frame.loc[frame["EventTime"] < TRAIN_END].copy()
    validation = frame.loc[
        (frame["EventTime"] >= TRAIN_END) & (frame["EventTime"] < TEST_START)
    ].copy()
    test = frame.loc[frame["EventTime"] >= TEST_START].copy()
    assigned = pd.concat([train, validation, test]).sort_values("SourceRow")
    if assigned["SourceRow"].tolist() != frame.sort_values("SourceRow")["SourceRow"].tolist():
        raise AssertionError("temporal masks are not a complete disjoint partition")
    return SplitData(train=train, validation=validation, test=test)


def split_manifest(frame: pd.DataFrame, source_hash: str) -> pd.DataFrame:
    split = temporal_split(frame)
    labels = pd.Series(index=frame.index, dtype="string")
    labels.loc[split.train.index] = "train"
    labels.loc[split.validation.index] = "validation"
    labels.loc[split.test.index] = "test"
    manifest = frame[["SourceRow", "UserID", "ProductID", "Timestamp", "EventTime"]].copy()
    manifest.insert(1, "SourceSHA256", source_hash)
    manifest["Split"] = labels
    return manifest.sort_values("SourceRow").reset_index(drop=True)


def audit_partitions(split: SplitData, catalog: pd.DataFrame) -> dict:
    train_users = set(split.train["UserID"])
    train_items = set(split.train["ProductID"])

    def holdout(name: str, frame: pd.DataFrame) -> dict:
        warm = frame["UserID"].isin(train_users)
        known = frame["ProductID"].isin(train_items)
        positive = frame["Rating"].ge(4)
        return {
            "name": name,
            "events": int(len(frame)),
            "users": int(frame["UserID"].nunique()),
            "items": int(frame["ProductID"].nunique()),
            "warm_user_events": int(warm.sum()),
            "cold_user_events": int((~warm).sum()),
            "known_item_events": int(known.sum()),
            "future_only_item_events": int((~known).sum()),
            "rating_error_eligible_events": int((warm & known).sum()),
            "ranking_eligible_users": int(frame.loc[known & positive, "UserID"].nunique()),
            "warm_ranking_eligible_users": int(frame.loc[warm & known & positive, "UserID"].nunique()),
            "cold_ranking_eligible_users": int(frame.loc[(~warm) & known & positive, "UserID"].nunique()),
            "future_only_items": sorted(frame.loc[~known, "ProductID"].unique().tolist()),
            "cold_users": sorted(frame.loc[~warm, "UserID"].unique().tolist()),
        }

    catalog_items = set(catalog["ProductID"])
    return {
        "train": {
            "events": int(len(split.train)),
            "users": int(split.train["UserID"].nunique()),
            "items": int(split.train["ProductID"].nunique()),
            "min_time": split.train["EventTime"].min().isoformat(),
            "max_time": split.train["EventTime"].max().isoformat(),
        },
        "validation": holdout("validation", split.validation),
        "test": holdout("test", split.test),
        "catalog": {
            "rows": int(len(catalog)),
            "matches_training_items": catalog_items == train_items,
            "missing_training_items": sorted(train_items - catalog_items),
            "extra_catalog_items": sorted(catalog_items - train_items),
        },
    }
