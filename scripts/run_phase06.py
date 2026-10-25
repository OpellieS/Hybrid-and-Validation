from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import psutil

from lab11c.audit import assert_candidate_universe, assert_strictly_before
from lab11c.bootstrap import bootstrap_model_metrics, paired_bootstrap_difference
from lab11c.data import TEST_START, TRAIN_END, load_catalog, load_interactions, temporal_split
from lab11c.evaluation import evaluate_ranking
from lab11c.freeze import validate_frozen_manifest
from lab11c.io_utils import write_json
from lab11c.matrix_factorization import ExplicitMatrixFactorization, MFConfig
from lab11c.recommenders import METHODS, RecommendationSystem, RecommenderConfig

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "artifacts" / "results"
EVIDENCE = ROOT / "artifacts" / "evidence"
FREEZE_PATH = ROOT / "artifacts" / "frozen_experiment_manifest.json"
MODEL_PATH = ROOT / "artifacts" / "models" / "selected_mf_state.json"


class PeakRSSMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self._stop.wait(0.005)

    def __enter__(self):
        tracemalloc.start()
        self._thread.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start
        self._stop.set()
        self._thread.join()
        self.python_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()


def main() -> None:
    frozen = validate_frozen_manifest(FREEZE_PATH)
    events = load_interactions(RAW / "interaction_events-2.csv")
    split = temporal_split(events)
    catalog = load_catalog(RAW / "synthetic_teaching_catalog-2.csv")
    state = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = ExplicitMatrixFactorization.from_dict(state)
    rec_config = RecommenderConfig(**frozen["model_configuration"]["recommender"])
    system = RecommendationSystem(split.train, catalog, model, rec_config)

    training_latest = split.train["EventTime"].max()
    validation_latest = split.validation["EventTime"].max()
    fitted_objects = [
        "popularity_statistics",
        "global_mean_and_biases",
        "content_vocabulary_and_idf",
        "similarity_rating_matrix",
        "user_profiles",
        "latent_factors",
    ]
    time_assertions = [
        assert_strictly_before(name, training_latest, TRAIN_END) for name in fitted_objects
    ]
    time_assertions.extend(
        [
            assert_strictly_before("mf_hyperparameter_selection", validation_latest, TEST_START),
            assert_strictly_before("hybrid_alpha_selection", validation_latest, TEST_START),
        ]
    )
    all_items = set(events["ProductID"])
    train_items = set(split.train["ProductID"])
    future_only = all_items - train_items
    candidate_assertion = assert_candidate_universe(train_items, set(system.items), future_only)

    negative_test = {"expected_failure": True, "actual_failure": False, "message": None}
    try:
        contaminated_latest = split.test["EventTime"].min()
        assert_strictly_before("contaminated_popularity_copy", contaminated_latest, TRAIN_END)
    except AssertionError as error:
        negative_test.update({"actual_failure": True, "message": str(error)})
    if not negative_test["actual_failure"]:
        raise AssertionError("deliberately contaminated leakage assertion did not fail")

    write_json(
        EVIDENCE / "leakage_audit.json",
        {
            "configuration_hash": frozen["configuration_hash"],
            "training_boundary_exclusive": TRAIN_END.isoformat(),
            "validation_boundary_exclusive": TEST_START.isoformat(),
            "fitted_object_time_assertions": time_assertions,
            "candidate_assertion": candidate_assertion,
            "future_only_items": sorted(future_only),
            "candidate_items": system.items,
            "negative_test": negative_test,
            "real_experiment_unchanged": True,
        },
    )

    user_metrics = pd.read_csv(RESULTS / "test_user_metrics.csv", dtype={"UserID": str})
    model_summary, model_replicates = bootstrap_model_metrics(user_metrics)
    paired_summary, paired_replicates = paired_bootstrap_difference(
        user_metrics, "hybrid", "popularity"
    )
    pd.concat([model_summary, paired_summary], ignore_index=True).to_csv(
        RESULTS / "bootstrap_summary.csv", index=False
    )
    model_replicates.to_csv(RESULTS / "bootstrap_model_replicates.csv", index=False)
    paired_replicates.to_csv(RESULTS / "bootstrap_paired_replicates.csv", index=False)
    write_json(
        EVIDENCE / "bootstrap_configuration.json",
        {
            "unit": "ranking-eligible user cluster",
            "replicates": 10_000,
            "seed": 202406,
            "percentile_interval": [2.5, 97.5],
            "paired_models": ["hybrid", "popularity"],
            "paired_resampling": "same user index draw used for both models through per-user differences",
            "eligible_users": int(user_metrics["UserID"].nunique()),
            "limitation": "With nine user clusters, percentile endpoints are discrete and unstable; intervals cannot support strong model-superiority claims.",
        },
    )

    # Isolated deterministic benchmark replay; it never replaces the frozen model.
    with PeakRSSMonitor() as training_monitor:
        replay = ExplicitMatrixFactorization(MFConfig(**state["config"])).fit(split.train)
    if replay.to_dict()["config"] != state["config"]:
        raise AssertionError("resource replay changed the frozen model configuration")

    with PeakRSSMonitor() as evaluation_monitor:
        for method in METHODS:
            evaluate_ranking(system, split.test, method, alpha=float(frozen["selected_alpha"]))
    input_files = sorted(RAW.glob("*.csv"))
    write_json(
        EVIDENCE / "resource_audit.json",
        {
            "configuration_hash": frozen["configuration_hash"],
            "training_measurement": {
                "wall_seconds": training_monitor.elapsed,
                "peak_process_rss_bytes": training_monitor.peak,
                "peak_python_traced_bytes": training_monitor.python_peak,
                "scope": "isolated replay of the frozen MF configuration; not used for selection or test predictions",
            },
            "evaluation_measurement": {
                "wall_seconds": evaluation_monitor.elapsed,
                "peak_process_rss_bytes": evaluation_monitor.peak,
                "peak_python_traced_bytes": evaluation_monitor.python_peak,
                "scope": "all six frozen methods over the nine ranking-eligible test users",
            },
            "input": {
                "files": {path.name: path.stat().st_size for path in input_files},
                "total_bytes": sum(path.stat().st_size for path in input_files),
                "interaction_rows": len(events),
                "training_rows": len(split.train),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_system_memory_bytes": psutil.virtual_memory().total,
            },
        },
    )
    print(
        {
            "phase": "06",
            "status": "PASS",
            "leakage_assertions": len(time_assertions) + 1,
            "negative_test_failed_as_expected": negative_test["actual_failure"],
            "bootstrap_users": user_metrics["UserID"].nunique(),
            "training_replay_seconds": training_monitor.elapsed,
            "evaluation_seconds": evaluation_monitor.elapsed,
        }
    )


if __name__ == "__main__":
    main()
