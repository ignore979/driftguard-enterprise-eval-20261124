"""Shared C artifact-selection chat contract for the SDK and PyTRiO."""

from __future__ import annotations

import json


C_SELECTION_SYSTEM = (
    "Select a C engine remaining-life artifact using the measured evidence. Return only JSON with "
    "keys select and reason. select is candidate or parent. Choose candidate only when eligible "
    "is true and host_vote is not reject. Otherwise choose parent. Use only supplied measurements; "
    "never claim hidden scores."
)


def select_messages(payload: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": C_SELECTION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]
