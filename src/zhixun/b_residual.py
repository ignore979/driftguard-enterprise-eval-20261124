"""Function-preserving residual basis migration for the official B MLP."""

from __future__ import annotations

import copy

import numpy as np

from zhixun.vendor.model import Model


def append_residual_basis(model: Model, x: np.ndarray) -> Model:
    """Append train-anchored tanh units with zero output and zero Adam moments."""
    if model.meta.get("profile") != "B" or model.meta["task"] != "regression":
        raise ValueError("Residual continuation requires the B regression checkpoint")
    x = model.normalized(x)
    feature_count = x.shape[1]
    original = model.meta["hidden"]
    added = 3 * feature_count + 1
    if original + added > 128 or len(x) == 0:
        raise ValueError("Residual migration exceeds the official width limit")
    quantiles = np.quantile(x, [.15, .5, .85], axis=0)
    spread = np.maximum((quantiles[-1] - quantiles[0]) * .25, 1e-5)
    slopes = np.tile(1.0 / spread, 3)
    incoming = np.zeros((feature_count, added), dtype=np.float64)
    incoming[np.tile(np.arange(feature_count), 3), np.arange(added - 1)] = slopes
    bias = np.concatenate((-quantiles.ravel() * slopes, [1.0]))
    migrated = copy.deepcopy(model)
    migrated.params["w1"] = np.column_stack((model.params["w1"], incoming))
    migrated.params["b1"] = np.concatenate((model.params["b1"], bias))
    migrated.params["w2"] = np.row_stack((model.params["w2"], np.zeros((added, 1))))
    for moments in (migrated.first, migrated.second):
        moments["w1"] = np.pad(moments["w1"], ((0, 0), (0, added)))
        moments["b1"] = np.pad(moments["b1"], (0, added))
        moments["w2"] = np.pad(moments["w2"], ((0, added), (0, 0)))
    migrated.meta["hidden"] = original + added
    migrated.meta["parameter_count"] = sum(p.size for p in migrated.params.values())
    migrated.meta["width_migration"] = {
        "from_hidden": original, "to_hidden": original + added,
        "prediction_preservation": "append_zero_output_basis",
        "adam_transfer": "original_prefix_unchanged_new_zero",
    }
    migrated.validate()
    return migrated


def fit_residual_outputs(model: Model, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                         *, parent_width: int, penalty: float) -> None:
    """Estimate only new output weights using public training split regression targets."""
    if (not 0 < parent_width < model.meta["hidden"] or not np.isfinite(penalty)
            or penalty <= 0 or weights.shape != y.shape or y.shape != (len(x),)
            or not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Invalid aligned targets, weights or residual options")
    hidden, _ = model.forward(model.normalized(x))
    baseline = (hidden[:, :parent_width] @ model.params["w2"][:parent_width] + model.params["b2"])[:, 0]
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"] - baseline
    basis = hidden[:, parent_width:]
    weighted = basis * weights[:, None]
    gram = weighted.T @ basis
    gram.flat[:: len(gram) + 1] += weights.sum() * penalty
    gram[-1, -1] -= weights.sum() * penalty  # The final constant centers the residuals.
    model.params["w2"][parent_width:, 0] = np.linalg.solve(gram, weighted.T @ target)
    model.validate()
