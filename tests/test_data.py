from pathlib import Path

import pandas as pd
import pytest

from lab11c.data import (
    audit_partitions,
    file_sha256,
    load_catalog,
    load_interactions,
    temporal_split,
)

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"


def test_main_source_and_temporal_contract():
    source = RAW / "interaction_events-2.csv"
    frame = load_interactions(source)
    split = temporal_split(frame)
    catalog = load_catalog(RAW / "synthetic_teaching_catalog-2.csv")
    audit = audit_partitions(split, catalog)

    assert file_sha256(source).upper() == "F55471CADB4740A4E15187BEA6B8A7B8143F7C9AC3C764269031348FCFBCC079"
    assert len(frame) == 200
    assert frame["UserID"].nunique() == 87
    assert frame["ProductID"].nunique() == 90
    assert frame["EventTime"].min() == pd.Timestamp("2024-03-09 19:14:56", tz="UTC")
    assert frame["EventTime"].max() == pd.Timestamp("2024-07-03 03:17:47", tz="UTC")
    assert [len(split.train), len(split.validation), len(split.test)] == [121, 38, 41]
    assert audit["train"]["items"] == 70
    assert audit["catalog"]["matches_training_items"] is True
    assert audit["validation"]["ranking_eligible_users"] == 10
    assert audit["validation"]["warm_ranking_eligible_users"] == 4
    assert audit["test"]["ranking_eligible_users"] == 9


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["1,10,4,1710000000", "1,10,5,1710000100"], "duplicate user-item"),
        (["1,10,4.5,1710000000"], "integer text"),
        (["1,10,6,1710000000"], r"in \[1, 5\]"),
        ([",10,4,1710000000"], "missing or blank"),
        (["1,10,4,tomorrow"], "Unix-second"),
    ],
)
def test_strict_ingestion_rejects_invalid_sources(tmp_path, rows, message):
    path = tmp_path / "bad.csv"
    path.write_text("UserID,ProductID,Rating,Timestamp\n" + "\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_interactions(path)
