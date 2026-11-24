"""One bounded A continuation using the measured robust gas recipe."""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from zhixun.a_gas_runtime import fit_gas_head
from zhixun.data import Dataset
from zhixun.model_extension import ClassWeightedModel
from zhixun.vendor.model import Model

SCHEDULE = (
    (1, .0005), (1, .0005), (1, .0005), (2, .0007), (3, .0007),
    (5, .001), (7, .001), (10, .001), (5, .0005), (10, .0005),
)
CW_POWER = 1.5


def train_candidate(parent: Model, training: Dataset, round_index: int, deadline: float) -> tuple[Model, list, dict]:
    """Copy the parent, continue its Adam state, and fit the readout on training only.

    The deadline is an absolute monotonic time. TimeoutError prevents the caller
    from adopting any partially trained candidate.
    """
    if type(round_index) is not int or not 1 <= round_index <= len(SCHEDULE):
        raise ValueError("A continuation requires a round in the measured schedule")
    if parent.meta.get("profile") != "A" or parent.meta["task"] != "classification":
        raise ValueError("A continuation requires a gas classification checkpoint")
    if not len(training.y) or time.monotonic() >= deadline:
        raise TimeoutError("A continuation has no trainable rows or budget")
    epochs, rate = SCHEDULE[round_index - 1]
    candidate = copy.deepcopy(parent)
    candidate.__class__ = ClassWeightedModel
    candidate.meta["learning_rate"] = rate
    losses = []
    for _ in range(epochs):
        if time.monotonic() >= deadline:
            raise TimeoutError("A continuation exceeded the training deadline")
        losses.extend(candidate.fit(training.x, training.y, 1, cw_power=CW_POWER))
    if time.monotonic() >= deadline:
        raise TimeoutError("A continuation exceeded the training deadline")
    candidate, readout = fit_gas_head(candidate, training.x, training.y)
    if time.monotonic() >= deadline:
        raise TimeoutError("A relative readout exceeded the training deadline")
    added_steps = candidate.meta["steps"] - parent.meta["steps"]
    if (
        not np.array_equal(candidate.mean, parent.mean)
        or not np.array_equal(candidate.scale, parent.scale)
        or candidate.meta["paths"] != parent.meta["paths"]
        or candidate.meta["classes"] != parent.meta["classes"]
        or candidate.meta["epochs"] - parent.meta["epochs"] != epochs
        or added_steps != epochs * math.ceil(len(training.y) / parent.meta["batch_size"])
    ):
        raise ValueError("A continuation changed the parent contract or optimizer lineage")
    candidate.validate()
    audit = {
        "strategy": readout["scheme"], "epochs": epochs, "learning_rate": rate,
        "cw_power": CW_POWER, "optimizer_steps_added": added_steps, "gas_readout": readout,
    }
    return candidate, losses, audit
