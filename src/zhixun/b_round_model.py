"""Input-only B continuation and quality checks in the native CPU checkpoint format."""
from __future__ import annotations

import copy
from itertools import combinations
import json
import math
from time import monotonic

import numpy as np

from zhixun.b_residual_tail import fit_asymmetric_residual
from zhixun.vendor.model import Model


NEAR_FAILURE_MINUTES = 100.0
RIDGE_PENALTY = .015
TAIL_MULTIPLIER = 4.
MAX_WIDTH = 128
RANDOM_SEED = 9204
RATIO_FEATURES = ("band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k", "rms")


def _ratio_pairs(parent: Model) -> list[tuple[int, int]]:
    """Identify documented physical channels by their feature paths, never row IDs."""
    index = {}
    for i, item in enumerate(parent.meta["paths"]):
        try:
            path = json.loads(item)
        except json.JSONDecodeError:
            continue
        if not isinstance(path, list) or not all(
            isinstance(part, list) and len(part) == 2 and part[0] == "key"
            and isinstance(part[1], str) for part in path
        ):
            continue
        names = [part[1] for part in path]
        if len(names) == 3 and names[0] == "features":
            index[tuple(names[1:])] = i
    required = {(channel, feature) for channel in ("horizontal", "vertical") for feature in RATIO_FEATURES}
    if not required <= index.keys():
        return []
    within = [
        pair for channel in ("horizontal", "vertical")
        for pair in combinations((index[channel, feature] for feature in RATIO_FEATURES[:3]), 2)
    ]
    across = [(index["horizontal", feature], index["vertical", feature]) for feature in RATIO_FEATURES]
    return within + across


def _append_projection_basis(parent: Model, x: np.ndarray, pairs: list[tuple[int, int]]) -> Model:
    """Add train-anchored amplitude, ratio and interaction tanh units at zero output."""
    z = parent.normalized(x)
    width = parent.meta["hidden"]
    feature_count = z.shape[1]
    free = MAX_WIDTH - width - 1  # Last unit supplies the unrestricted residual intercept.
    low, q1, q3, high = np.quantile(z, [.15, .25, .75, .85], axis=0)
    spread = np.maximum(.25 * (high - low), 1e-5)
    directions, offsets = [], []

    def add(direction, bias=0.):
        directions.append(direction)
        offsets.append(bias)

    for quantile in (q1, q3):
        for feature in range(feature_count):
            vector = np.zeros(feature_count)
            vector[feature] = 1. / spread[feature]
            add(vector, -quantile[feature] / spread[feature])
    for numerator, denominator in pairs:
        ratio = np.maximum(x[:, numerator], 0.) / np.maximum(x[:, denominator], 1e-8)
        threshold = float(np.median(ratio))
        # Amplitude ratios become an affine decision in the original normalized feature coordinates.
        vector = np.zeros(feature_count)
        vector[numerator] = parent.scale[numerator]
        vector[denominator] = -threshold * parent.scale[denominator]
        raw_bias = parent.mean[numerator] - threshold * parent.mean[denominator]
        a, b = np.quantile(z @ vector, [.2, .8])
        norm = max((b - a) * .11, 1e-8)
        add(vector / norm, raw_bias / norm)
    rng = np.random.default_rng(RANDOM_SEED)
    while len(directions) < free:
        vector = np.zeros(feature_count)
        indexes = rng.choice(feature_count, min(6, feature_count), replace=False)
        vector[indexes] = rng.normal(size=len(indexes)) / spread[indexes]
        a, center, b = np.quantile(z @ vector, [.15, .5, .85])
        norm = max((b - a) * .28, 1e-6)
        add(vector / norm, -center / norm)
    add(np.zeros(feature_count), 1.)
    added = len(directions)
    candidate = copy.deepcopy(parent)
    candidate.params["w1"] = np.column_stack((parent.params["w1"], np.asarray(directions).T))
    candidate.params["b1"] = np.r_[parent.params["b1"], offsets]
    candidate.params["w2"] = np.vstack((parent.params["w2"], np.zeros((added, 1))))
    for moments in (candidate.first, candidate.second):
        moments["w1"] = np.pad(moments["w1"], ((0, 0), (0, added)))
        moments["b1"] = np.pad(moments["b1"], (0, added))
        moments["w2"] = np.pad(moments["w2"], ((0, added), (0, 0)))
    candidate.meta["hidden"] = width + added
    candidate.meta["parameter_count"] = sum(p.size for p in candidate.params.values())
    candidate.meta["b_projection_residual"] = {
        "parent_hidden": width, "basis": "amplitude_ratio_sparse_tanh", "seed": RANDOM_SEED,
    }
    candidate.validate()
    return candidate


def continue_b(parent: Model, x: np.ndarray, y: np.ndarray, *, deadline: float = math.inf,
               step: float = 1.) -> tuple[Model, list[float], dict]:
    """Fit an asymmetric ridge residual using only the host's training inputs and RUL targets.

    The returned list contains convex solver objective values, not Adam epochs.
    The native NumPy forward path, input scaler and prior Adam counters remain unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (parent.meta.get("profile") != "B" or parent.meta["task"] != "regression"
            or x.ndim != 2 or x.shape[1] != len(parent.meta["paths"])
            or y.shape != (len(x),) or not len(y) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any(y < 0)
            or not math.isfinite(step) or not 0 < step <= 1
            or math.isnan(deadline)):
        raise ValueError("Invalid B projection continuation")
    if monotonic() >= deadline:
        raise TimeoutError("B training budget elapsed")

    audit = {"strategy": "asymmetric_projection_residual", "ridge_penalty": RIDGE_PENALTY,
             "near_failure_minutes": NEAR_FAILURE_MINUTES, "overestimate_multiplier": TAIL_MULTIPLIER,
             "near_failure_training_rows": int(np.sum(y < NEAR_FAILURE_MINUTES)), "residual_step": step,
             "adam_epochs_completed": 0, "migration_max_abs_minutes": 0.,
             "hidden_width": parent.meta["hidden"], "solver_iterations": 0}
    state = parent.meta.get("b_projection_residual")
    ratios = _ratio_pairs(parent)
    if state is None:
        if not ratios or parent.meta["hidden"] + 2 * x.shape[1] + len(ratios) + 1 > MAX_WIDTH:
            return copy.deepcopy(parent), [], {**audit, "migration": "unsupported_schema_or_width"}
        width = parent.meta["hidden"]
        candidate = _append_projection_basis(parent, x, ratios)
        delta = float(np.max(np.abs(candidate.predict(x) - parent.predict(x))))
        if delta > 1e-9:
            raise ValueError("B residual migration changed the parent function")
        audit.update(migration="amplitude_ratio_sparse_tanh", migration_max_abs_minutes=delta)
    else:
        width = state.get("parent_hidden") if isinstance(state, dict) else None
        if (not ratios or not isinstance(state, dict) or state.get("basis") != "amplitude_ratio_sparse_tanh"
                or type(width) is not int or not 0 < width < parent.meta["hidden"]):
            raise ValueError("Invalid B residual checkpoint metadata")
        candidate = copy.deepcopy(parent)
        audit["migration"] = "reuse_b_projection_basis"
    if monotonic() >= deadline:
        raise TimeoutError("B training budget elapsed after migration")
    baseline = candidate.params["w2"][width:, 0].copy()
    trace = fit_asymmetric_residual(
        candidate, x, y, np.ones_like(y), parent_width=width, penalty=RIDGE_PENALTY,
        multiplier=TAIL_MULTIPLIER,
    )
    candidate.params["w2"][width:, 0] = baseline + step * (candidate.params["w2"][width:, 0] - baseline)
    if monotonic() >= deadline:
        raise TimeoutError("B training budget elapsed during residual fitting")
    candidate.validate()
    audit["hidden_width"] = candidate.meta["hidden"]
    audit["solver_iterations"] = len(trace)
    return candidate, trace, audit


def measure_b(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Compute host-safe measurements from paired nonnegative RUL values."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (prediction.shape != target.shape or target.ndim != 1 or not len(target)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()
            or np.any(prediction < 0) or np.any(target < 0)):
        raise ValueError("B measurements require aligned, finite and nonnegative RUL vectors")
    error = prediction - target

    def metrics(delta):
        positive = delta[delta > 0]
        return {
            "rows": len(delta),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "mae": float(np.mean(np.abs(delta))),
            "bias": float(np.mean(delta)),
            "overestimate_p95": float(np.quantile(positive, .95)) if len(positive) else 0.,
            "overestimate_fraction": float(np.mean(delta > 0)),
        }

    near = error[target < NEAR_FAILURE_MINUTES]
    return {"overall": metrics(error), "near_failure": metrics(near) if len(near) else None}


def promote_b(origin: dict, parent: dict, candidate: dict) -> dict:
    """Require marginal improvement and preserve the initial near-failure error bounds."""
    initial, before, after = (value["overall"] for value in (origin, parent, candidate))
    near_initial, near_before, near_after = (value["near_failure"] for value in (origin, parent, candidate))
    if initial["rows"] != before["rows"] or before["rows"] != after["rows"]:
        raise ValueError("B candidate must use the same public panel as its parent")
    support = [value["rows"] if value is not None else 0 for value in (near_initial, near_before, near_after)]
    if len(set(support)) != 1:
        raise ValueError("B candidate must have the same near-failure validation support")
    epsilon = 1e-9
    checks = {
        "rmse_improved_vs_parent": after["rmse"] < before["rmse"] - epsilon,
        "mae_not_worse_vs_parent": after["mae"] <= before["mae"] + epsilon,
    }
    if near_initial is None:
        checks["overestimate_tail_not_worse_vs_origin"] = (
            after["overestimate_p95"] <= initial["overestimate_p95"] + epsilon
        )
    else:
        checks.update({
            "near_failure_rmse_not_worse_vs_origin": near_after["rmse"] <= near_initial["rmse"] + epsilon,
            "near_failure_overestimate_tail_not_worse_vs_origin": (
                near_after["overestimate_p95"] <= near_initial["overestimate_p95"] + epsilon
            ),
        })
    return {"pass": all(checks.values()), "checks": checks}
