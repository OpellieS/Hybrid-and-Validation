from __future__ import annotations

from dataclasses import asdict
from itertools import product
from pathlib import Path

import pandas as pd

from lab11c.data import load_interactions, temporal_split
from lab11c.io_utils import object_sha256, write_json
from lab11c.matrix_factorization import ExplicitMatrixFactorization, MFConfig, rating_errors

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
EVIDENCE = ROOT / "artifacts" / "evidence"
MODELS = ROOT / "artifacts" / "models"

# Predeclared before validation scoring because the instructor companion is absent.
GRID = {
    "factors": [2, 4, 8],
    "learning_rate": [0.2, 0.5],
    "regularization": [0.001, 0.01],
    "epochs": [300],
    "seed": [202411],
}


def main() -> None:
    split = temporal_split(load_interactions(RAW / "interaction_events-2.csv"))
    records = []
    states = {}
    for factors, rate, regularization, epochs, seed in product(
        GRID["factors"], GRID["learning_rate"], GRID["regularization"], GRID["epochs"], GRID["seed"]
    ):
        config = MFConfig(factors, rate, regularization, epochs, seed)
        model = ExplicitMatrixFactorization(config).fit(split.train)
        metrics = rating_errors(model, split.validation)
        key = object_sha256(asdict(config))[:12]
        states[key] = model.to_dict()
        records.append(
            {
                "candidate_id": key,
                **asdict(config),
                "validation_events": metrics["events"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "initial_objective": model.history[0]["objective"],
                "final_objective": model.history[-1]["objective"],
            }
        )
    record = pd.DataFrame(records).sort_values(
        ["validation_rmse", "validation_mae", "factors", "regularization", "learning_rate", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    record["selected"] = False
    record.loc[0, "selected"] = True
    selected_id = str(record.loc[0, "candidate_id"])
    selected_state = states[selected_id]

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    record.to_csv(EVIDENCE / "mf_validation_search.csv", index=False)
    write_json(EVIDENCE / "mf_predeclared_grid.json", GRID)
    write_json(MODELS / "selected_mf_state.json", selected_state)
    write_json(
        EVIDENCE / "mf_selection.json",
        {
            "selection_metric": "minimum validation RMSE on warm-user and training-known-item rating events",
            "tie_break": ["validation MAE", "fewer factors", "lower regularization", "lower learning rate", "candidate ID"],
            "selected_candidate_id": selected_id,
            "selected_config": selected_state["config"],
            "validation_metrics": {
                "events": int(record.loc[0, "validation_events"]),
                "mae": float(record.loc[0, "validation_mae"]),
                "rmse": float(record.loc[0, "validation_rmse"]),
            },
            "model_state_sha256": object_sha256(selected_state),
            "test_rows_accessed": False,
        },
    )
    update = ExplicitMatrixFactorization.one_event_update(
        4.0, 3.0, 0.2, -0.1, [0.2, 0.4], [0.5, 0.1], 0.05, 0.1
    )
    write_json(EVIDENCE / "mf_one_event_update.json", update)
    print(
        {
            "phase": "02",
            "status": "PASS",
            "candidates": len(record),
            "selected": selected_state["config"],
            "validation_rmse": float(record.loc[0, "validation_rmse"]),
        }
    )


if __name__ == "__main__":
    main()
