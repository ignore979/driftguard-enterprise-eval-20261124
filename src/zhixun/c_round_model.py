"""Input-only C continuation suitable for repeated SDK rounds."""
from __future__ import annotations

import copy
import math
from time import monotonic

import numpy as np

from zhixun.c_model import condition_residual, engine_weights, fit_epoch
from zhixun.vendor.model import Model

OPERATING_SETTING = '[["key","settings"],["index",0]]'
KNOWN_REGIMES = frozenset((0, 10, 20, 25, 35, 42))
C_STRATEGIES = frozenset(("plain", "condition", "tail"))


def continue_c(
    parent: Model, x: np.ndarray, y: np.ndarray, *, epochs: int, learning_rate: float,
    strategy: str, deadline: float = math.inf,
) -> tuple[Model, list[float], dict]:
    """Continue a copied parent without access to engine IDs or validation rows."""
    if (parent.meta.get("profile") != "C" or strategy not in C_STRATEGIES
            or type(epochs) is not int or not 1 <= epochs <= 10
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= .001
            or x.ndim != 2 or x.shape[1] != len(parent.meta["paths"])
            or y.shape != (len(x),) or not len(y) or not np.isfinite(y).all()
            or np.any(y < 0) or not np.isfinite(x).all()):
        raise ValueError("Invalid C round continuation")

    audit = {"strategy": strategy, "migration": "none", "regimes": [], "architecture": parent.meta["hidden"]}
    model = copy.deepcopy(parent)
    weights = None
    if strategy != "plain" and OPERATING_SETTING in parent.meta["paths"]:
        column = x[:, parent.meta["paths"].index(OPERATING_SETTING)]
        rounded = np.rint(column)
        distinct = frozenset(rounded.tolist())
        if 2 <= len(distinct) <= 6 and distinct <= KNOWN_REGIMES and np.max(np.abs(column - rounded)) <= .05:
            regimes = rounded.astype(np.int64)
            audit["regimes"] = sorted(set(regimes.tolist()))
            # A single dummy group yields .8 per-row plus .2 per-regime loss.
            weights = engine_weights(np.zeros_like(regimes), regimes)
            if "regime_residual" not in parent.meta and parent.meta["hidden"] + 16 * len(distinct) <= 128:
                if monotonic() >= deadline:
                    raise TimeoutError("C round budget elapsed before migration")
                model, delta = condition_residual(parent, x, y, weights)
                audit.update(migration="condition_residual", migration_max_abs_cycles=delta)
            elif "regime_residual" not in parent.meta:
                audit["migration"] = "width_limit"
            else:
                audit["migration"] = "reuse_parent_condition_basis"
        else:
            audit["migration"] = "single_or_unrecognized_regime_standard_fallback"
    elif strategy != "plain":
        audit["migration"] = "missing_setting_standard_fallback"
    losses = []
    model.meta["learning_rate"] = float(learning_rate)
    for _ in range(epochs):
        if monotonic() >= deadline:
            raise TimeoutError("C round training budget elapsed")
        if weights is None:
            losses.extend(model.fit(x, y, 1))
        else:
            losses.append(fit_epoch(model, x, y, weights, near_weight=1.5 if strategy == "tail" else 1.,
                                    tail_weight=.18 if strategy == "tail" else 0.))
    if monotonic() >= deadline:
        raise TimeoutError("C round overran its training budget")
    count = math.ceil(len(y) / parent.meta["batch_size"])
    if (model.meta["epochs"] != parent.meta["epochs"] + epochs
            or model.meta["steps"] != parent.meta["steps"] + count * epochs
            or not np.array_equal(model.mean, parent.mean) or not np.array_equal(model.scale, parent.scale)
            or model.meta["target_mean"] != parent.meta["target_mean"]
            or model.meta["target_scale"] != parent.meta["target_scale"]):
        raise ValueError("C round violated checkpoint/scaler ancestry")
    audit["architecture"] = model.meta["hidden"]
    return model, losses, audit


def promote_c(origin: dict, parent: dict, candidate: dict) -> dict:
    """Compare marginal RMSE/MAE and preserve initial near-failure error bounds."""
    base, old, new = (metrics["overall"] for metrics in (origin, parent, candidate))
    near_base, near_new = (metrics["lifetime"]["0-50"] for metrics in (origin, candidate))
    conditions = {
        "rmse_improved_vs_parent": new["rmse"] < old["rmse"],
        "mae_not_worse_vs_parent": new["mae"] <= old["mae"],
        "nasa_not_worse_vs_origin": new["nasa_reference_sum"] <= base["nasa_reference_sum"],
        "near_failure_rmse_not_worse_vs_origin": near_new["rmse"] <= near_base["rmse"],
        "near_failure_p95_not_worse_vs_origin": near_new["overestimate_p95"] <= near_base["overestimate_p95"],
    }
    return {"pass": all(conditions.values()), "checks": conditions}
