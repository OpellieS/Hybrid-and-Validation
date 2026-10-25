"""Deterministic explicit-rating matrix factorization with full-batch gradients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MFConfig:
    factors: int
    learning_rate: float
    regularization: float
    epochs: int
    seed: int
    init_std: float = 0.05


class ExplicitMatrixFactorization:
    """Fit r_ui = mu + b_u + b_i + p_u dot q_i on observed training rows.

    The optimized objective is

        0.5 / N * sum_observed (r_ui - rhat_ui)^2
        + 0.5 * lambda * (||b_u||^2 + ||b_i||^2 + ||P||^2 + ||Q||^2).

    Each epoch is one simultaneous full-batch gradient step. Therefore every
    gradient is computed from the same old parameter state. With N=1 this
    reduces exactly to the assignment's printed one-event update.
    """

    def __init__(self, config: MFConfig):
        self.config = config
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "ExplicitMatrixFactorization":
        if train.empty:
            raise ValueError("training data cannot be empty")
        self.user_ids = sorted(train["UserID"].astype(str).unique().tolist())
        self.item_ids = sorted(train["ProductID"].astype(str).unique().tolist())
        self.user_to_index = {value: index for index, value in enumerate(self.user_ids)}
        self.item_to_index = {value: index for index, value in enumerate(self.item_ids)}
        self.mu = float(train["Rating"].mean())
        self.training_max_time = train["EventTime"].max().isoformat()

        rng = np.random.default_rng(self.config.seed)
        self.user_bias = np.zeros(len(self.user_ids), dtype=float)
        self.item_bias = np.zeros(len(self.item_ids), dtype=float)
        self.user_factors = rng.normal(0.0, self.config.init_std, (len(self.user_ids), self.config.factors))
        self.item_factors = rng.normal(0.0, self.config.init_std, (len(self.item_ids), self.config.factors))

        user_index = train["UserID"].map(self.user_to_index).to_numpy(int)
        item_index = train["ProductID"].map(self.item_to_index).to_numpy(int)
        ratings = train["Rating"].to_numpy(float)
        n = len(ratings)
        self.history: list[dict[str, float | int]] = []

        for epoch in range(self.config.epochs):
            old_user_bias = self.user_bias.copy()
            old_item_bias = self.item_bias.copy()
            old_user_factors = self.user_factors.copy()
            old_item_factors = self.item_factors.copy()

            predictions = (
                self.mu
                + old_user_bias[user_index]
                + old_item_bias[item_index]
                + np.einsum("ij,ij->i", old_user_factors[user_index], old_item_factors[item_index])
            )
            residual = ratings - predictions

            grad_user_bias = self.config.regularization * old_user_bias
            grad_item_bias = self.config.regularization * old_item_bias
            grad_user_factors = self.config.regularization * old_user_factors
            grad_item_factors = self.config.regularization * old_item_factors
            np.add.at(grad_user_bias, user_index, -residual / n)
            np.add.at(grad_item_bias, item_index, -residual / n)
            np.add.at(
                grad_user_factors,
                user_index,
                -(residual[:, None] * old_item_factors[item_index]) / n,
            )
            np.add.at(
                grad_item_factors,
                item_index,
                -(residual[:, None] * old_user_factors[user_index]) / n,
            )

            rate = self.config.learning_rate
            self.user_bias = old_user_bias - rate * grad_user_bias
            self.item_bias = old_item_bias - rate * grad_item_bias
            self.user_factors = old_user_factors - rate * grad_user_factors
            self.item_factors = old_item_factors - rate * grad_item_factors

            if not all(
                np.isfinite(array).all()
                for array in (self.user_bias, self.item_bias, self.user_factors, self.item_factors)
            ):
                raise FloatingPointError("matrix-factorization parameters became non-finite")
            if epoch in {0, self.config.epochs - 1} or (epoch + 1) % 50 == 0:
                self.history.append(
                    {
                        "epoch": epoch + 1,
                        "objective": self._encoded_objective(user_index, item_index, ratings),
                    }
                )

        self.fitted = True
        return self

    def _encoded_objective(self, user_index: np.ndarray, item_index: np.ndarray, ratings: np.ndarray) -> float:
        predictions = (
            self.mu
            + self.user_bias[user_index]
            + self.item_bias[item_index]
            + np.einsum("ij,ij->i", self.user_factors[user_index], self.item_factors[item_index])
        )
        squared_error = 0.5 * float(np.mean(np.square(ratings - predictions)))
        penalty = 0.5 * self.config.regularization * float(
            np.square(self.user_bias).sum()
            + np.square(self.item_bias).sum()
            + np.square(self.user_factors).sum()
            + np.square(self.item_factors).sum()
        )
        return squared_error + penalty

    def objective(self, observed: pd.DataFrame) -> float:
        self._require_fitted()
        if not set(observed["UserID"]).issubset(self.user_to_index) or not set(observed["ProductID"]).issubset(self.item_to_index):
            raise ValueError("objective requires IDs present in the fitted state")
        users = observed["UserID"].map(self.user_to_index).to_numpy(int)
        items = observed["ProductID"].map(self.item_to_index).to_numpy(int)
        return self._encoded_objective(users, items, observed["Rating"].to_numpy(float))

    def predict_one(self, user_id: str, item_id: str, clip: bool = False) -> float:
        self._require_fitted()
        score = self.mu
        user_index = self.user_to_index.get(str(user_id))
        item_index = self.item_to_index.get(str(item_id))
        if user_index is not None:
            score += self.user_bias[user_index]
        if item_index is not None:
            score += self.item_bias[item_index]
        if user_index is not None and item_index is not None:
            score += float(np.dot(self.user_factors[user_index], self.item_factors[item_index]))
        return float(np.clip(score, 1.0, 5.0) if clip else score)

    def predict_frame(self, frame: pd.DataFrame, clip: bool = False) -> np.ndarray:
        return np.array(
            [self.predict_one(user, item, clip=clip) for user, item in zip(frame["UserID"], frame["ProductID"])],
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "class": type(self).__name__,
            "config": asdict(self.config),
            "mu": self.mu,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "user_bias": self.user_bias.tolist(),
            "item_bias": self.item_bias.tolist(),
            "user_factors": self.user_factors.tolist(),
            "item_factors": self.item_factors.tolist(),
            "training_max_time": self.training_max_time,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ExplicitMatrixFactorization":
        model = cls(MFConfig(**state["config"]))
        model.mu = float(state["mu"])
        model.user_ids = [str(value) for value in state["user_ids"]]
        model.item_ids = [str(value) for value in state["item_ids"]]
        model.user_to_index = {value: index for index, value in enumerate(model.user_ids)}
        model.item_to_index = {value: index for index, value in enumerate(model.item_ids)}
        model.user_bias = np.asarray(state["user_bias"], dtype=float)
        model.item_bias = np.asarray(state["item_bias"], dtype=float)
        model.user_factors = np.asarray(state["user_factors"], dtype=float)
        model.item_factors = np.asarray(state["item_factors"], dtype=float)
        model.training_max_time = str(state["training_max_time"])
        model.history = list(state.get("history", []))
        model._assert_dimensions()
        model.fitted = True
        return model

    @staticmethod
    def one_event_update(
        rating: float,
        mu: float,
        user_bias: float,
        item_bias: float,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        learning_rate: float,
        regularization: float,
    ) -> dict[str, Any]:
        old_p = np.asarray(user_factors, dtype=float).copy()
        old_q = np.asarray(item_factors, dtype=float).copy()
        prediction = float(mu + user_bias + item_bias + np.dot(old_p, old_q))
        residual = float(rating - prediction)
        return {
            "prediction": prediction,
            "residual": residual,
            "user_bias": float(user_bias + learning_rate * (residual - regularization * user_bias)),
            "item_bias": float(item_bias + learning_rate * (residual - regularization * item_bias)),
            "user_factors": old_p + learning_rate * (residual * old_q - regularization * old_p),
            "item_factors": old_q + learning_rate * (residual * old_p - regularization * old_q),
        }

    def _assert_dimensions(self) -> None:
        if self.user_bias.shape != (len(self.user_ids),):
            raise ValueError("user-bias dimension mismatch")
        if self.item_bias.shape != (len(self.item_ids),):
            raise ValueError("item-bias dimension mismatch")
        if self.user_factors.shape != (len(self.user_ids), self.config.factors):
            raise ValueError("user-factor dimension mismatch")
        if self.item_factors.shape != (len(self.item_ids), self.config.factors):
            raise ValueError("item-factor dimension mismatch")

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("model has not been fitted")
        self._assert_dimensions()


def rating_errors(model: ExplicitMatrixFactorization, frame: pd.DataFrame) -> dict[str, float | int]:
    eligible = frame[
        frame["UserID"].isin(model.user_to_index) & frame["ProductID"].isin(model.item_to_index)
    ]
    if eligible.empty:
        return {"events": 0, "mae": float("nan"), "rmse": float("nan")}
    truth = eligible["Rating"].to_numpy(float)
    prediction = model.predict_frame(eligible, clip=True)
    error = truth - prediction
    return {
        "events": int(len(eligible)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }
