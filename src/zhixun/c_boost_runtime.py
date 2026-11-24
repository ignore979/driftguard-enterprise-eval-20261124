"""Train-only C residual with bounded cycle corrections."""

from __future__ import annotations

import base64
import copy
import hashlib
import time
import zlib

import numpy as np

from zhixun.vendor.model import Model

HEAD_KEY = "c_bounded_boost"
SCHEME = "lightgbm_residual_tanh"
TREES = 160
MAX_TEXT_BYTES = 2 * 1024 * 1024
CAP_CYCLES = 12.
SOFT_SCALE = 28.
SETTINGS = {
    "objective": "regression", "verbosity": -1, "seed": 2026, "deterministic": True,
    "force_col_wise": True, "num_threads": 1, "learning_rate": .06, "num_leaves": 31,
    "max_depth": 7, "min_data_in_leaf": 180, "lambda_l2": 12, "max_bin": 63,
}


class BoundedBoostModel(Model):
    def predict(self, x):
        base = super().predict(x)
        if not len(base):
            return base
        residual = self.booster.predict(self.normalized(x).astype(np.float32), num_threads=1)
        if residual.shape != base.shape or not np.isfinite(residual).all():
            raise ValueError("Invalid C residual output")
        correction = CAP_CYCLES * np.tanh(residual / SOFT_SCALE)
        return np.maximum(0., base + correction)


def _check_profile(model: Model) -> None:
    if model.meta.get("profile") != "C" or model.meta["task"] != "regression":
        raise ValueError("Bounded residual requires the official C regression task")


def _attach(model: Model, booster) -> Model:
    if booster.num_feature() != len(model.meta["paths"]) or not 1 <= booster.num_trees() <= TREES:
        raise ValueError("Invalid C residual architecture")
    model.booster = booster
    model.__class__ = BoundedBoostModel
    return model


def enable_bounded_boost(model: Model) -> Model:
    """Restore the optional head from the validated CPU checkpoint."""
    if HEAD_KEY not in model.meta or isinstance(model, BoundedBoostModel):
        return model
    _check_profile(model)
    payload = model.meta[HEAD_KEY]
    if (not isinstance(payload, dict) or set(payload) != {"scheme", "state", "sha256"}
            or payload["scheme"] != SCHEME or not isinstance(payload["state"], str)
            or len(payload["state"]) > MAX_TEXT_BYTES):
        raise ValueError("Invalid C bounded residual schema")
    compressed = base64.b64decode(payload["state"], validate=True)
    decoder = zlib.decompressobj()
    native = decoder.decompress(compressed, MAX_TEXT_BYTES + 1)
    if (len(native) > MAX_TEXT_BYTES or not decoder.eof or decoder.unused_data
            or hashlib.sha256(native).hexdigest() != payload["sha256"]):
        raise ValueError("Invalid C bounded residual checkpoint")
    import lightgbm

    return _attach(model, lightgbm.Booster(model_str=native.decode("utf-8")))


def fit_bounded_boost(model: Model, x: np.ndarray, y: np.ndarray, *, deadline: float) -> tuple[Model, dict]:
    """Fit the residual against the unchanged C base using the training projection."""
    _check_profile(model)
    if HEAD_KEY in model.meta:
        raise ValueError("The C residual can only be fitted once per base checkpoint")
    if y.shape != (len(x),) or not len(y) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Invalid C bounded residual training targets")

    def enforce_deadline(_=None):
        if time.monotonic() >= deadline:
            raise TimeoutError("C residual exceeded the training time budget")

    enforce_deadline()
    import lightgbm

    dataset = lightgbm.Dataset(model.normalized(x).astype(np.float32), label=y - model.predict(x))
    booster = lightgbm.train(SETTINGS, dataset, num_boost_round=TREES, callbacks=[enforce_deadline])
    enforce_deadline()
    native = booster.model_to_string().encode("utf-8")
    if len(native) > MAX_TEXT_BYTES:
        raise ValueError("C residual exceeds the checkpoint limit")
    encoded = base64.b64encode(zlib.compress(native)).decode("ascii")
    if len(encoded) > MAX_TEXT_BYTES:
        raise ValueError("Encoded C residual exceeds the checkpoint limit")
    extended = copy.deepcopy(model)
    checksum = hashlib.sha256(native).hexdigest()
    extended.meta[HEAD_KEY] = {"scheme": SCHEME, "state": encoded, "sha256": checksum}
    return _attach(extended, booster), {
        "strategy": SCHEME, "trees": booster.num_trees(), "training_rows": len(y),
        "cap_cycles": CAP_CYCLES, "soft_scale": SOFT_SCALE, "residual_sha256": checksum,
    }


def boost_checks(reference: dict, candidate: dict) -> dict[str, bool]:
    """Require a joint improvement before exposing a tree-adjusted candidate."""
    old, new = reference["overall"], candidate["overall"]
    checks = {"rmse": new["rmse"] < old["rmse"]}
    for metric in ("mae", "nasa_reference_sum", "overestimate_p95"):
        checks[metric] = new[metric] <= old[metric]
    old_near, new_near = reference["lifetime"]["0-50"], candidate["lifetime"]["0-50"]
    checks["near_failure_observed"] = old_near is not None and new_near is not None
    if checks["near_failure_observed"]:
        for metric in ("rmse", "overestimate_p95"):
            checks["near_failure_" + metric] = new_near[metric] <= old_near[metric]
    return checks
