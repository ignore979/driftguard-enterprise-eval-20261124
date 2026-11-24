"""Workspace-v2 continuation for C, with the frozen path for other tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from zhixun.c_model import c_metrics
from zhixun.c_round import measure_c, quality_gate
from zhixun.c_round_model import continue_c, promote_c
from zhixun.contest_runtime import PREDICT_ARGV, TASKS, predict, rows_dataset, train as frozen_train
from zhixun.c_trio_protocol import select_messages
from zhixun.data import digest, write_json
from zhixun.vendor.model import Model

DEFAULT_C_STRATEGY = "condition"
C_SCHEDULE = ((8, .001), (8, .0007), (8, .001), (8, .0005), (8, .0007),
              (5, .0005), (5, .00035), (5, .00025), (5, .00015), (5, .0001))


def host_vote(receipt: dict, before: dict, after: dict) -> str:
    """Use the measured RMSE when the host receipt can be reconciled to the local panel."""
    summary = receipt.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    value = metrics.get("rmse") if isinstance(metrics, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unavailable"
    if not math.isclose(value, after["rmse"], rel_tol=1e-5, abs_tol=1e-4):
        return "unavailable"
    return "accept" if value < before["rmse"] - 1e-6 else "reject"


def parse_choice(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        return "parent", "Missing model selection text."
    try:
        value = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    except ValueError:
        return "parent", "Model selection was not valid JSON."
    if (not isinstance(value, dict) or set(value) != {"select", "reason"}
            or value["select"] not in ("parent", "candidate")
            or not isinstance(value["reason"], str) or not value["reason"].strip()):
        return "parent", "Model selection did not satisfy the action schema."
    return value["select"], value["reason"].strip()[:240]


def _previous_choices(state: Path, round_index: int) -> list[dict]:
    history = []
    for index in range(max(1, round_index - 3), round_index):
        path = state / f"round-{index:02d}.json"
        if path.is_file():
            entry = json.loads(path.read_text(encoding="utf-8"))
            history.append({"round": index, "select": entry["select"], "recipe": entry["recipe"],
                            "candidate_rmse": entry["local_public_after"]["rmse"]})
    return history


def _panel_revision(data) -> str:
    hasher = hashlib.sha256(np.ascontiguousarray(data.x).tobytes())
    hasher.update(np.ascontiguousarray(data.y).tobytes())
    return hasher.hexdigest()


def _guard_panel(prediction, target) -> dict:
    near = target < 50
    return {"overall": c_metrics(prediction, target),
            "lifetime": {"0-50": c_metrics(prediction[near], target[near]) if near.any() else None}}


def _origin_for_panel(state, revision, guard_parent):
    path = state / "public-panel.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    origin = old["origin"] if old.get("revision") == revision else guard_parent
    write_json(path, {"revision": revision, "origin": origin})
    return origin


def train(rnd):
    if rnd.dataset_key != "C":
        return frozen_train(rnd)
    start = time.monotonic()
    work = Path(os.environ.get("CONTEST_WORK", "/workspace")) / f"enterprise-c-r{rnd.round_index}"
    work.mkdir(parents=True, exist_ok=False)
    history = rnd.history()
    for relative in rnd.parent.descriptor["code_paths"]:
        rnd.parent.path(relative)
    checkpoints = [name for name in rnd.parent.descriptor["checkpoint_paths"] if name.endswith(".npz")]
    if len(checkpoints) != 1:
        raise ValueError("C requires exactly one declared CPU NPZ parent")
    parent_path = rnd.parent.path(checkpoints[0])
    parent = Model.load(parent_path)
    if parent.meta.get("profile") != "C" or parent.meta["task"] != rnd.task["task"]:
        raise ValueError("C parent does not match the host task")
    training = rows_dataset(rnd.data.training_rows(), parent)
    validation = rows_dataset(rnd.data.validation_rows(), parent)
    if set(training.ids) & set(validation.ids):
        raise ValueError("Training and public validation IDs overlap")
    parent_output = parent.predict(validation.x)
    before = measure_c(parent_output, validation.y)
    guard_parent = _guard_panel(parent_output, validation.y)
    state = rnd.state_workspace(work / "state")
    revision = _panel_revision(validation)
    origin = _origin_for_panel(state, revision, guard_parent)
    compact_history = _previous_choices(state, rnd.round_index)
    epochs, rate = C_SCHEDULE[min(rnd.round_index, len(C_SCHEDULE)) - 1]
    strategy = DEFAULT_C_STRATEGY
    recipe_config = {"strategy": strategy, "epochs": epochs, "rate": rate}
    allowance = min(180., max(0., (rnd.wall_seconds or 300.) - (time.monotonic() - start) - 15.))
    hypothesis = "C continuation should reduce measured RMSE while containing asymmetric and near-end errors."
    reason = ""
    submitted = None
    losses = []
    migration_delta = 0.
    parent_sha256 = digest(parent_path)
    checkpoint_sha256 = None
    try:
        candidate, losses, training_audit = continue_c(
            parent, training.x, training.y, epochs=epochs, learning_rate=rate,
            strategy=strategy, deadline=time.monotonic() + allowance,
        )
    except TimeoutError:
        after = before
        training_audit = {"migration": "timeout"}
        eligible, violations = False, ["train_timeout"]
        change = "Training exceeded its budget; retained the parent without publishing partial weights."
    else:
        migration_delta = training_audit.get("migration_max_abs_cycles", 0.)
        output = candidate.predict(validation.x)
        after = measure_c(output, validation.y)
        guard_candidate = _guard_panel(output, validation.y)
        if guard_candidate["lifetime"]["0-50"] is None:
            eligible, violations = quality_gate(before, after)
        else:
            gate = promote_c(origin, guard_parent, guard_candidate)
            eligible = gate["pass"]
            violations = [name for name, passed in gate["checks"].items() if not passed]
        change = (
            f"Continued the host parent for {len(losses)} epochs at {rate:g}; "
            f"width {parent.meta['hidden']} to {candidate.meta['hidden']}; method {strategy}."
        )
        out = work / "candidate"
        out.mkdir()
        file = out / "checkpoint.npz"
        candidate.save(file)
        checkpoint_sha256 = digest(file)
        write_json(out / "experiment.json", {
            "recipe": recipe_config, "parent_sha256": parent_sha256, "migration_delta": migration_delta,
            "training_audit": training_audit, "public_panel_revision": revision,
            "validation_before": before, "validation_after": after, "losses": losses,
            "eligible": eligible, "violations": violations, "checkpoint_sha256": checkpoint_sha256,
        })
        submitted = rnd.submit_candidate(
            out, hypothesis=hypothesis, change_summary=change, seed=parent.meta["seed"],
            configuration={"recipe": recipe_config, "parent_sha256": parent_sha256, "completed_epochs": len(losses)},
            checkpoint_paths=["checkpoint.npz"], code_paths=["experiment.json"], predict_argv=PREDICT_ARGV,
        )
    vote = host_vote(submitted, before, after) if submitted is not None else "unavailable"
    payload = {
        "dataset": "C", "round": rnd.round_index, "parent": before, "candidate": after,
        "eligible": eligible, "violations": violations, "host_vote": vote,
        "recipe": strategy, "history": compact_history,
    }
    messages = select_messages(payload)
    answer = rnd.model(messages, max_tokens=min(160, rnd.max_output_tokens), temperature=0, enable_thinking=False)
    choice, reason = parse_choice(answer.text)
    adopt = submitted is not None and eligible and vote != "reject" and choice == "candidate"
    selected = submitted["artifact_id"] if adopt else rnd.parent.artifact_id
    if not adopt and violations:
        reason = f"{reason} Local guard: {', '.join(violations)}."
    if not adopt and vote == "reject":
        reason += " Host RMSE did not improve."
    audit = {
        "dataset": "C", "round": rnd.round_index, "recipe": strategy, "recipe_configuration": recipe_config,
        "training_audit": training_audit, "public_panel_revision": revision,
        "parent_sha256": parent_sha256, "candidate_sha256": checkpoint_sha256,
        "selected_artifact_id": selected, "select": "candidate" if adopt else "parent",
        "local_public_before": before, "local_public_after": after, "eligible": eligible,
        "violations": violations, "losses": losses, "migration_delta": migration_delta,
        "history_ids": [item["id"] for item in history], "model_call_id": answer.call_id,
        "model_messages": messages, "model_response": answer.text, "reason": reason,
        "host_candidate_receipt": submitted,
    }
    write_json(state / f"round-{rnd.round_index:02d}.json", audit)
    rnd.save_state(state)
    rnd.finish(
        selected_artifact_id=selected, hypothesis=hypothesis, change_summary=change, reason=reason,
        configuration={"recipe": recipe_config, "guard_passed": eligible, "host_vote": vote},
        seed=parent.meta["seed"], history_ids=list(rnd.history_ids), model_call_ids=list(rnd.call_ids),
    )
