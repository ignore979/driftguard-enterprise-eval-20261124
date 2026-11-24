"""Train-only asymmetric residual ridge for the standard B checkpoint."""

from __future__ import annotations

import numpy as np

from zhixun.b_residual import fit_residual_outputs
from zhixun.vendor.model import Model


def fit_asymmetric_residual(model: Model, x: np.ndarray, target: np.ndarray, weights: np.ndarray,
                            *, parent_width: int, penalty: float, multiplier: float) -> list[float]:
    """Penalize positive residual errors for training RUL below 100 minutes."""
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("The positive error multiplier must be finite and at least one")
    fit_residual_outputs(model, x, target, weights, parent_width=parent_width, penalty=penalty)
    hidden, _ = model.forward(model.normalized(x))
    baseline = (hidden[:, :parent_width] @ model.params["w2"][:parent_width] + model.params["b2"])[:, 0]
    residual_target = (target - model.meta["target_mean"]) / model.meta["target_scale"] - baseline
    basis = hidden[:, parent_width:]
    near_failure = (0 <= target) & (target < 100)
    weights = weights / weights.sum()
    regularization = np.full(basis.shape[1], penalty)
    regularization[-1] = 0.0
    coeff = model.params["w2"][parent_width:, 0].copy()

    def error_and_weights(values):
        error = basis @ values - residual_target
        active_weights = np.where(near_failure & (error > 0), multiplier, 1) * weights
        return error, active_weights

    def objective(values, error, active_weights):
        return float(np.dot(active_weights, error ** 2) + np.dot(regularization, values ** 2))

    trace = []
    for _ in range(30):
        error, active_weights = error_and_weights(coeff)
        current = objective(coeff, error, active_weights)
        trace.append(current)
        gradient = basis.T @ (active_weights * error) + regularization * coeff
        if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            model.params["w2"][parent_width:, 0] = coeff
            model.validate()
            return trace
        hessian = (basis * active_weights[:, None]).T @ basis + np.diag(regularization)
        direction = -np.linalg.solve(hessian, gradient)
        directional_derivative = 2 * np.dot(gradient, direction)
        for exponent in range(25):
            step = 0.5 ** exponent
            update = coeff + step * direction
            error, active_weights = error_and_weights(update)
            if objective(update, error, active_weights) <= current + 1e-4 * step * directional_derivative:
                coeff = update
                break
        else:
            raise ValueError("Asymmetric ridge line search failed")
    raise ValueError("Asymmetric ridge did not converge")
