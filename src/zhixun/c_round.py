"""C continuation and quality checks for the official row projection."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np

from zhixun.data import Dataset
from zhixun.model_extension import expand_hidden
from zhixun.vendor.model import Model


@dataclass(frozen=True)
class Recipe:
    name: str
    width: int
    epochs: int
    rate: float
    positive_weight: float
    late_weight: float


EARLY = Recipe("asymmetric-wide", 96, 8, .001, 1.8, 1.15)
LATE = Recipe("asymmetric-refine", 96, 6, .0005, 1.35, 1.1)
REFRESH = Recipe("symmetric-refine", 96, 5, .0006, 1.0, 1.0)


def round_recipe(index: int) -> Recipe:
    if type(index) is not int or index < 1:
        raise ValueError("A positive C round number is required")
    if index <= 3:
        return EARLY
    return REFRESH if index % 3 == 0 else LATE


def measure_c(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (prediction.ndim != 1 or prediction.shape != target.shape or not len(target)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()
            or np.any(prediction < 0) or np.any(target < 0)):
        raise ValueError("Expected aligned finite nonnegative RUL vectors")
    error = prediction - target
    exponent = np.maximum(error / 10, -error / 13)
    high = float(np.max(exponent))
    log_mean = max(0., high + float(np.log(np.exp(exponent - high).mean())))
    positive = error[error > 0]
    near = error[target < 30]
    return {
        "rows": len(error),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
        "nasa_log_mean": log_mean,
        "overestimate_p95": float(np.quantile(positive, .95)) if len(positive) else 0.,
        "near_end_rmse": float(np.sqrt(np.square(near).mean())) if len(near) else None,
        "near_end_bias": float(near.mean()) if len(near) else None,
    }


def nasa_sum(metrics: dict) -> float:
    """Convert the stable selection metric for panels where the raw sum is representable."""
    return math.expm1(metrics["nasa_log_mean"]) * metrics["rows"]


def quality_gate(parent: dict, candidate: dict) -> tuple[bool, list[str]]:
    if parent["rows"] != candidate["rows"]:
        raise ValueError("The C gate requires the same validation panel")
    limits = {"mae": .05, "nasa_log_mean": 1e-10, "overestimate_p95": 1e-8, "near_end_rmse": .5}
    violations = ["rmse"] if candidate["rmse"] >= parent["rmse"] - 1e-6 else []
    for key, slack in limits.items():
        a, b = parent[key], candidate[key]
        if (a is None) != (b is None):
            raise ValueError("C near-end support changed")
        if a is not None and b > a + slack:
            violations.append(key)
    a, b = parent["near_end_bias"], candidate["near_end_bias"]
    if (a is None) != (b is None):
        raise ValueError("C near-end support changed")
    if a is not None and max(0., b) > max(0., a) + .5:
        violations.append("near_end_bias")
    return not violations, violations


def fit_c_epoch(model: Model, normalized: np.ndarray, target: np.ndarray,
                late: np.ndarray, recipe: Recipe, deadline: float) -> float:
    seed, epoch = model.meta["seed"], model.meta["epochs"]
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(target))
    batch_size = model.meta["batch_size"]
    loss_sum = 0.
    for offset in range(0, len(order), batch_size):
        if offset % (64 * batch_size) == 0 and time.monotonic() >= deadline:
            raise TimeoutError("C training budget exhausted")
        indices = order[offset:offset + batch_size]
        batch = normalized[indices]
        hidden, logits = model.forward(batch)
        error = logits[:, 0] - target[indices]
        # Preserve the parent objective on underestimates. Weight optimism, especially near failure.
        weight = np.where(error > 0, recipe.positive_weight, 1.)
        weight *= np.where((error > 0) & late[indices], recipe.late_weight, 1.)
        loss_sum += float(np.sum(error ** 2 * weight))
        delta = (2 * error * weight / len(indices))[:, None]
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
        hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
        gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("Nonfinite continuation gradient")
        model.meta["steps"] += 1
        step = model.meta["steps"]
        for name, gradient in gradients.items():
            gradient *= min(1.0, 5.0 / max(norm, 1e-12))
            model.first[name] = .9 * model.first[name] + .1 * gradient
            model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
            first = model.first[name] / (1 - .9 ** step)
            second = model.second[name] / (1 - .999 ** step)
            model.params[name] -= recipe.rate * first / (np.sqrt(second) + 1e-8)
    model.meta["epochs"] += 1
    return loss_sum / len(target)


def continue_c(parent: Model, data: Dataset, recipe: Recipe, deadline: float) -> tuple[Model, list[float], float]:
    if (parent.meta.get("profile") != "C" or parent.meta["task"] != "regression"
            or data.y.shape != (len(data.x),) or not len(data.y) or not np.isfinite(data.y).all()
            or np.any(data.y < 0) or type(recipe.epochs) is not int or not 1 <= recipe.epochs <= 10
            or not 0 < recipe.rate <= .001 or not 1 <= recipe.positive_weight <= 4
            or not 1 <= recipe.late_weight <= 2):
        raise ValueError("Invalid C continuation settings or data")
    width = max(parent.meta["hidden"], recipe.width)
    candidate = expand_hidden(parent, width) if width != parent.meta["hidden"] else copy.deepcopy(parent)
    sample = data.x[:min(len(data.x), 1024)]
    migration_delta = float(np.abs(parent.predict(sample) - candidate.predict(sample)).max())
    if migration_delta > 1e-8:
        raise ValueError("C hidden-layer migration changed the starting prediction")
    candidate.meta["learning_rate"] = recipe.rate
    normalized = candidate.normalized(data.x)
    target = (data.y - candidate.meta["target_mean"]) / candidate.meta["target_scale"]
    losses = [
        fit_c_epoch(candidate, normalized, target, data.y < 30, recipe, deadline)
        for _ in range(recipe.epochs)
    ]
    if time.monotonic() >= deadline:
        raise TimeoutError("C training budget exhausted")
    if (candidate.meta["epochs"] != parent.meta["epochs"] + recipe.epochs
            or candidate.meta["steps"] != parent.meta["steps"] + recipe.epochs * math.ceil(
                len(data.y) / parent.meta["batch_size"]
            )):
        raise ValueError("C optimizer counters did not advance correctly")
    candidate.validate()
    return candidate, losses, migration_delta
