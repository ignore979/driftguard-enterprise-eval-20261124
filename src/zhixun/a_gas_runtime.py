"""Scale-stable gas readout with the official MLP and checkpoint format retained."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zhixun.metrics import promotion
from zhixun.model_extension import ClassWeightedModel
from zhixun.vendor.model import Model

HEAD_KEY = "a_gas_head"
SCHEME = "relative_shape_class_balanced_ridge"
FEATURES, CLASSES = 128, 6
RIDGE = .03
ROBUST_SHARE = .65
REFERENCE_MARGIN = 5.


@dataclass(frozen=True)
class GasReadout:
    scale: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    coefficients: np.ndarray
    temperature: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        features = np.clip((_shape(x, self.scale) - self.center) / self.spread, -8., 8.)
        return (features @ self.coefficients[:-1] + self.coefficients[-1]) / self.temperature


def _shape(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Divide out a row-wide gain before feature scaling to avoid overflow on large inputs.
    magnitude = np.max(np.abs(x), axis=1, keepdims=True)
    relative = x / np.where(magnitude > 0, magnitude, 1.) / scale
    length = np.linalg.norm(relative, axis=1, keepdims=True)
    return relative / np.where(length > 0, length, 1.)


def _check_model(model: Model) -> None:
    if (
        model.meta.get("profile") != "A"
        or model.meta["task"] != "classification"
        or len(model.meta["classes"]) != CLASSES
        or len(model.meta["paths"]) != FEATURES
    ):
        raise ValueError("Gas readout requires the official A task and its unchanged feature contract")


class GasReadoutModel(ClassWeightedModel):
    def predict(self, x):
        original_logits = self.forward(self.normalized(x))[1]
        logits = (1. - ROBUST_SHARE) * original_logits + ROBUST_SHARE * self.gas_readout.logits(x)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def enable_gas_head(model: Model) -> Model:
    """Restore the optional gas readout after loading the ordinary NPZ checkpoint."""
    if HEAD_KEY not in model.meta:
        return model
    _check_model(model)
    payload = model.meta[HEAD_KEY]
    fields = {"scheme", "scale", "center", "spread", "coefficients", "temperature"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["scheme"] != SCHEME:
        raise ValueError("Invalid gas readout schema")
    shapes = {"scale": (FEATURES,), "center": (FEATURES,), "spread": (FEATURES,),
              "coefficients": (FEATURES + 1, CLASSES)}
    arrays = {}
    for name, shape in shapes.items():
        array = np.asarray(payload[name], dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid gas readout {name}")
        arrays[name] = array
    temperature = payload["temperature"]
    if (
        np.any(arrays["scale"] < 1e-8) or np.any(arrays["spread"] < 1e-6)
        or not isinstance(temperature, (int, float)) or not np.isfinite(temperature)
        or temperature < 1e-6
    ):
        raise ValueError("Invalid gas readout scaling")
    model.gas_readout = GasReadout(**arrays, temperature=float(temperature))
    model.__class__ = GasReadoutModel
    return model


def fit_gas_head(model: Model, x: np.ndarray, y: np.ndarray) -> tuple[Model, dict]:
    """Fit compact class-balanced ridge coefficients from authorized training data only.

    Mutates the supplied candidate metadata. Parameter arrays, optimizer state,
    parent feature paths and normalization are left as received from continuation.
    """
    _check_model(model)
    if (
        x.ndim != 2 or x.shape[1] != FEATURES or y.shape != (len(x),)
        or not np.isfinite(x).all() or y.dtype.kind not in "iu"
        or np.any(y < 0) or np.any(y >= CLASSES)
    ):
        raise ValueError("Invalid gas readout training data")
    counts = np.bincount(y, minlength=CLASSES)
    audit = {"scheme": SCHEME, "ridge": RIDGE, "robust_share": ROBUST_SHARE,
             "train_rows": len(x), "class_support": counts.tolist()}
    # Keep a previously fitted readout if the round has an incomplete class projection.
    if np.any(counts < 3):
        return enable_gas_head(model), {**audit, "fit": "insufficient_class_support"}
    median_abs = np.median(np.abs(x), axis=0)
    if not np.any(median_abs > 0):
        return enable_gas_head(model), {**audit, "fit": "degenerate_scale"}
    floor = max(float(np.median(median_abs[median_abs > 0])) * .0001, 1e-8)
    scale = np.maximum(median_abs, floor)
    relative = _shape(x, scale)
    center = np.median(relative, axis=0)
    spread = np.maximum(np.std(relative, axis=0), 1e-6)
    design = np.c_[np.clip((relative - center) / spread, -8., 8.), np.ones(len(x))]
    weighted = design.T / counts[y]
    penalty = np.eye(FEATURES + 1) * RIDGE
    penalty[-1, -1] = 1e-8
    coefficients = np.linalg.solve(weighted @ design + penalty, weighted @ np.eye(CLASSES)[y])
    logits = np.sort(design @ coefficients, axis=1)
    temperature = max(float(np.median(logits[:, -1] - logits[:, -2])) / REFERENCE_MARGIN, 1e-6)
    model.meta[HEAD_KEY] = {
        "scheme": SCHEME, "scale": scale.tolist(), "center": center.tolist(),
        "spread": spread.tolist(), "coefficients": coefficients.tolist(), "temperature": temperature,
    }
    return enable_gas_head(model), {**audit, "fit": "trained"}


def gas_promotion(before: dict, after: dict, classes_before: dict, classes_after: dict) -> tuple[bool, str]:
    """Require improvement with no accuracy regression or newly missing detected class."""
    accepted, rule = promotion(before, after, "A")
    collapsed = [
        name for name, metrics in classes_before.items()
        if isinstance(metrics, dict) and metrics.get("support", 0) > 0
        and metrics.get("recall", 0) > 0 and classes_after[name]["recall"] == 0
    ]
    if collapsed:
        return False, "previously detected classes must not collapse: " + ", ".join(collapsed)
    return accepted, rule
