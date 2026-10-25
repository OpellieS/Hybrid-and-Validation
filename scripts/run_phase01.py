from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab11c.data import (
    TEST_START,
    TRAIN_END,
    audit_partitions,
    file_sha256,
    load_catalog,
    load_interactions,
    split_manifest,
    temporal_split,
)
from lab11c.io_utils import write_json
from lab11c.teaching import (
    centered_overlap_similarity,
    cosine,
    manual_tfidf,
    observed_mask,
    observed_mean,
    rating_weighted_profile,
    term_counts,
)

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
EVIDENCE = ROOT / "artifacts" / "evidence"


def main() -> None:
    source = RAW / "interaction_events-2.csv"
    source_hash = file_sha256(source)
    events = load_interactions(source)
    catalog = load_catalog(RAW / "synthetic_teaching_catalog-2.csv")
    split = temporal_split(events)
    audit = audit_partitions(split, catalog)
    audit.update(
        {
            "source_sha256": source_hash,
            "schema": {column: str(dtype) for column, dtype in events.dtypes.items()},
            "rows": len(events),
            "unique_users": events["UserID"].nunique(),
            "unique_products": events["ProductID"].nunique(),
            "duplicate_user_item_pairs": int(events.duplicated(["UserID", "ProductID"]).sum()),
            "missing_required_values": int(events[["UserID", "ProductID", "Rating", "Timestamp"]].isna().sum().sum()),
            "rating_values": sorted(events["Rating"].unique().astype(int).tolist()),
            "time_min": events["EventTime"].min().isoformat(),
            "time_max": events["EventTime"].max().isoformat(),
            "train_end_exclusive": TRAIN_END.isoformat(),
            "test_start_inclusive": TEST_START.isoformat(),
        }
    )
    write_json(EVIDENCE / "data_audit.json", audit)
    split_manifest(events, source_hash).to_csv(EVIDENCE / "split_manifest.csv", index=False)

    source_hashes = {path.name: file_sha256(path) for path in sorted(RAW.glob("*.csv"))}
    write_json(EVIDENCE / "source_hashes.json", source_hashes)

    tiny_ratings = pd.read_csv(RAW / "tiny_ratings-2.csv").set_index("UserID").to_numpy(float)
    tiny_catalog = pd.read_csv(RAW / "tiny_content_catalog-2.csv")
    documents = tiny_catalog["Description"].tolist()
    vocabulary, vectors, idf = manual_tfidf(documents)
    means = np.nanmean(tiny_ratings, axis=1)
    cf_similarity, cf_overlap = centered_overlap_similarity(
        tiny_ratings[0], tiny_ratings[1], means[0], means[1]
    )
    profile = rating_weighted_profile(vectors[:3], np.array([5.0, 3.0, 4.0]))
    teaching = {
        "declared_variant": {
            "tokenization": "lowercase whitespace tokens",
            "tf": "0 if count=0 else 1+ln(count)",
            "idf": "ln(number_of_documents/document_frequency), no smoothing or additive constant",
            "normalization": "row L2",
            "profile": "absolute-rating-weighted average of L2 item vectors, then L2 normalize",
            "cf_similarity": "cosine of global-mean-centered ratings on co-rated entries",
            "reason": "Week 12 deck and companion script were unavailable",
        },
        "observed_mask": observed_mask(tiny_ratings).astype(int),
        "observed_count": int(observed_mask(tiny_ratings).sum()),
        "global_observed_mean": float(observed_mean(tiny_ratings)),
        "user_observed_means": observed_mean(tiny_ratings, axis=1),
        "term_counts": [dict(term_counts(document)) for document in documents],
        "vocabulary": vocabulary,
        "idf": idf,
        "tfidf_l2": vectors,
        "row_l2_norms": np.linalg.norm(vectors, axis=1),
        "cosine_A_B": cosine(vectors[0], vectors[1]),
        "cosine_A_C": cosine(vectors[0], vectors[2]),
        "rating_weighted_profile_A_B_C": profile,
        "cf_U0_U1_overlap": cf_overlap,
        "cf_U0_U1_similarity": cf_similarity,
    }
    write_json(EVIDENCE / "teaching_examples.json", teaching)
    print(json.dumps({"phase": "01", "status": "PASS", "split_sizes": [len(split.train), len(split.validation), len(split.test)], "training_items": split.train["ProductID"].nunique(), "test_ranking_users": audit["test"]["ranking_eligible_users"]}))


if __name__ == "__main__":
    main()
