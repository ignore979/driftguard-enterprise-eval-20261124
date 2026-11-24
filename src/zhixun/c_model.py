"""C engine RUL continuation in the native CPU checkpoint format."""
from __future__ import annotations

import copy
import math

import numpy as np

from zhixun.vendor.model import Model


def engine_weights(engine: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """Give smaller engines and operating conditions explicit weight in the objective."""
    def inverse_size(labels):
        _, groups, count = np.unique(labels, return_inverse=True, return_counts=True)
        return len(labels) / (len(count) * count[groups])

    if engine.shape != regime.shape or engine.ndim != 1 or not len(engine):
        raise ValueError("Engine and regime groups must align")
    return .5 + .3 * inverse_size(engine) + .2 * inverse_size(regime)


def fit_epoch(
    model: Model, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
    *, near_weight: float = 1.0, tail_weight: float = 0.0,
) -> float:
    """Run one original Adam epoch with optional asymmetric regression loss."""
    if (
        model.meta.get("profile") != "C" or model.meta["task"] != "regression"
        or y.shape != (len(x),) or y.shape != weights.shape or not len(y)
        or not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0)
        or not np.isclose(weights.mean(), 1) or not 1 <= near_weight <= 5 or not 0 <= tail_weight <= 1
    ):
        raise ValueError("Invalid C continuation input")
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    normalized = model.normalized(x)
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
    indices = rng.permutation(len(x))
    loss = 0.0
    batch_size = model.meta["batch_size"]
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        batch = normalized[rows]
        hidden, raw = model.forward(batch)
        error = raw[:, 0] - target[rows]
        effective = np.where((y[rows] < 50) & (error > 0), near_weight, 1.0) * weights[rows]
        # A smooth extra tail penalty reduces pressure from very large raw score exponents.
        coefficient = tail_weight * np.where(error > 0, 1.3, 1.0)
        abs_error = np.abs(error)
        loss += float(np.sum(effective * error ** 2 * (1 + coefficient * abs_error)))
        delta = (effective * error * (2 + 3 * coefficient * abs_error) / len(rows))[:, None]
        hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                     "w1": batch.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
        gradient_norm = math.sqrt(sum(float(np.sum(v ** 2)) for v in gradients.values()))
        if not math.isfinite(gradient_norm):
            raise ValueError("C continuation produced a nonfinite gradient")
        model.meta["steps"] += 1
        step = model.meta["steps"]
        for name, gradient in gradients.items():
            gradient *= min(1.0, 5.0 / max(gradient_norm, 1e-12))
            model.first[name] = .9 * model.first[name] + .1 * gradient
            model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
            first = model.first[name] / (1 - .9 ** step)
            second = model.second[name] / (1 - .999 ** step)
            model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
    model.meta["epochs"] += 1
    model.validate()
    return loss / len(x)


def condition_residual(model: Model, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[Model, float]:
    """Continue the parent by fitting paired tanh bases for within-condition sensor trends."""
    setting_name = '[["key","settings"],["index",0]]'
    path = model.meta["paths"]
    setting = path.index(setting_name)
    sensors = np.array([idx for idx, item in enumerate(path) if '"sensors"' in item])
    regimes = np.rint(x[:, setting]).astype(int)
    known = np.unique(regimes)
    original = model.meta["hidden"]
    n_added = 16 * len(known)
    if model.meta.get("profile") != "C" or not 2 <= len(known) <= 6 or original + n_added > 128:
        raise ValueError("C residual expects the official operating conditions and checkpoint width")
    if y.shape != (len(x),) or y.shape != weights.shape or not np.isfinite(y).all() or not np.all(weights > 0):
        raise ValueError("C residual requires aligned training targets and positive weights")
    z = model.normalized(x)
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    columns, offsets = [], []
    for condition in known:
        local = regimes == condition
        outside = ~local
        center = z[local].mean(axis=0)
        scale = np.maximum(z[local][:, sensors].std(axis=0), .005)
        standardized = (z[local][:, sensors] - center[sensors]) / scale
        covariance = standardized.T @ standardized / local.sum()
        health = np.linalg.solve(
            covariance + .3 * np.eye(len(sensors)),
            standardized.T @ (target[local] - target[local].mean()) / local.sum(),
        )
        health /= max(float(np.std(standardized @ health)), .05)
        _, components = np.linalg.eigh(covariance)
        directions = [np.zeros_like(health)]
        for component in components[:, -3:].T:
            directions.append(component / max(float(np.std(standardized @ component)), .05))
        pairs = [(directions[0], .9), *((health, offset) for offset in (-1., -.3, .3, 1.)),
                 *((direction, .2) for direction in directions[1:])]
        gap = np.abs(z[outside, setting] - center[setting])
        for direction, bias in pairs:
            sensor_coef = direction / scale
            local_raw_outside = (z[outside][:, sensors] - center[sensors]) @ sensor_coef + bias
            # Outside the regime, both tanh units approach the same sign. Inside it,
            # their difference follows the local sensor projection.
            gain = max(20.0, float(np.max((np.abs(local_raw_outside) + 9.0) / gap)))
            gate = np.zeros(z.shape[1])
            gate[setting] = gain
            signal = gate.copy()
            signal[sensors] = sensor_coef
            columns.extend([signal, gate])
            offsets.extend([bias - center @ signal, -center @ gate])
    model_copy = copy.deepcopy(model)
    model_copy.params["w1"] = np.column_stack((model.params["w1"], *columns))
    model_copy.params["b1"] = np.r_[model.params["b1"], offsets]
    model_copy.params["w2"] = np.r_[model.params["w2"], np.zeros((n_added, 1))]
    for state in (model_copy.first, model_copy.second):
        state["w1"] = np.pad(state["w1"], ((0, 0), (0, n_added)))
        state["b1"] = np.pad(state["b1"], (0, n_added))
        state["w2"] = np.pad(state["w2"], ((0, n_added), (0, 0)))
    model_copy.meta["hidden"] += n_added
    model_copy.meta["parameter_count"] = sum(value.size for value in model_copy.params.values())
    model_copy.meta["regime_residual"] = {"added_neurons": n_added, "conditioning_setting": setting_name}
    model_copy.validate()
    migration_delta = float(np.max(np.abs(model_copy.predict(x) - model.predict(x))))
    if migration_delta > 1e-9:
        raise ValueError("Condition residual changed the initial parent function")
    hidden, baseline = model_copy.forward(z)
    extras = hidden[:, original:]
    basis = extras[:, ::2] - extras[:, 1::2]
    weighted = weights[:, None] * basis
    gram = weighted.T @ basis
    gram.flat[::len(gram) + 1] += .003 * weights.sum()
    coef = np.linalg.solve(gram, weighted.T @ (target - baseline[:, 0]))
    model_copy.params["w2"][original::2, 0] = coef
    model_copy.params["w2"][original + 1::2, 0] = -coef
    model_copy.validate()
    return model_copy, migration_delta


def c_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Measure cycles and a local exponential reference; platform NASA is separate."""
    error = prediction - target
    if error.shape != target.shape or error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError("Invalid aligned C predictions")
    positive = error[error > 0]
    nasa = np.expm1(np.where(error > 0, error / 10, -error / 13))
    if not np.isfinite(nasa).all():
        raise ValueError("Exponential C error reference overflowed")
    return {"rows": len(error), "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
            "signed_bias": float(error.mean()), "overestimate_fraction": float(np.mean(error > 0)),
            "overestimate_p95": float(np.quantile(positive, .95)) if len(positive) else 0.0,
            "nasa_reference_sum": float(nasa.sum()), "nasa_reference_mean": float(nasa.mean())}
