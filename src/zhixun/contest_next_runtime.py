"""SDK workspace-v2 continuation with per-task candidates and shared audit rules."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from zhixun.a_gas_runtime import enable_gas_head, gas_promotion
from zhixun.a_next_round import train_candidate as train_a
from zhixun.b_round_model import continue_b, measure_b, promote_b
from zhixun.c_boost_runtime import HEAD_KEY as C_HEAD_KEY, boost_checks, enable_bounded_boost, fit_bounded_boost
from zhixun.c_round import measure_c, quality_gate
from zhixun.c_round_model import continue_c, promote_c
from zhixun.contest_c_runtime import C_SCHEDULE, _guard_panel, parse_choice
from zhixun.contest_runtime import PREDICT_ARGV, TASKS, rows_dataset
from zhixun.data import digest, write_json
from zhixun.metrics import classification_metrics
from zhixun.vendor.data import flatten
from zhixun.vendor.model import Model

_MODELS: dict[Path, Model] = {}


def load_model(file: Path) -> Model:
    return enable_bounded_boost(enable_gas_head(Model.load(file)))


def measurements(task, model, panel):
    values = model.predict(panel.x)
    if task == "A":
        overall, classes = classification_metrics(values, panel.y, model.meta["classes"])
        return overall, classes
    if task == "B":
        result = measure_b(values, panel.y)
        return result["overall"], result
    return measure_c(values, panel.y), _guard_panel(values, panel.y)


def candidate_for(task, parent, training, index, deadline):
    if task == "A":
        return train_a(parent, training, index, deadline)
    if task == "B":
        model, _, info = continue_b(parent, training.x, training.y, deadline=deadline)
        return model, [], info
    epochs, rate = C_SCHEDULE[min(index, len(C_SCHEDULE)) - 1]
    return continue_c(parent, training.x, training.y, epochs=epochs, learning_rate=rate,
                      strategy="condition", deadline=deadline)


def decision(task, origin, before, after, before_detail, after_detail):
    if task == "A":
        eligible, rule = gas_promotion(before, after, before_detail, after_detail)
        return eligible, [] if eligible else [rule]
    if task == "B":
        check = promote_b(origin, before_detail, after_detail)
    elif after_detail["lifetime"]["0-50"] is None:
        return quality_gate(before, after)
    else:
        check = promote_c(origin, before_detail, after_detail)
    return check["pass"], [key for key, passed in check["checks"].items() if not passed]


def end_c_candidate(model, detail, parent, before_detail, eligible, training, validation, deadline):
    reference, reference_detail = (model, detail) if eligible else (parent, before_detail)
    if C_HEAD_KEY in reference.meta:
        return model, {"result": "already_extended"}
    try:
        updated, report = fit_bounded_boost(reference, training.x, training.y, deadline=deadline)
    except TimeoutError:
        return model, {"result": "deadline", "reference": "continuation" if eligible else "parent"}
    after, updated_detail = measurements("C", updated, validation)
    checks = boost_checks(reference_detail, updated_detail)
    use = all(checks.values())
    report.update(result="passed" if use else "guard_rejected", checks=checks,
                  reference="continuation" if eligible else "parent",
                  reference_metrics=reference_detail, candidate_metrics=updated_detail)
    return updated if use else model, report


def comparable_host_vote(task, receipt, before, after):
    if receipt is None:
        return "unavailable"
    metrics = receipt.get("summary", {}).get("metrics", {})
    main = "macro_f1" if task == "A" else "rmse"
    value = metrics.get(main)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unavailable"
    if not math.isclose(value, after[main], rel_tol=1e-5, abs_tol=1e-4):
        return "unavailable"
    return "accept" if (value > before[main] if task == "A" else value < before[main]) else "reject"


def reference_panel(state, validation, before):
    signature = hashlib.sha256(np.ascontiguousarray(validation.x).tobytes())
    signature.update(np.ascontiguousarray(validation.y).tobytes())
    revision = signature.hexdigest()
    file = state / "next-public-panel.json"
    previous = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
    origin = previous["origin"] if previous.get("revision") == revision else before
    write_json(file, {"revision": revision, "origin": origin})
    return origin, revision


def abbreviated_history(state, iteration, main):
    history = []
    for index in range(max(1, iteration - 3), iteration):
        audit = state / f"next-round-{index:02d}.json"
        if audit.is_file():
            data = json.loads(audit.read_text(encoding="utf-8"))
            history.append({"round": index, "select": data["select"], "candidate": data["after"][main]})
    return history


def train(rnd):
    started = time.monotonic()
    task = rnd.dataset_key
    if task not in {"A", "B", "C"}:
        raise ValueError("Unknown industrial task")
    iteration = rnd.round_index
    work = Path(os.environ.get("CONTEST_WORK", "/workspace")) / f"next-{task}-r{iteration}"
    work.mkdir(parents=True, exist_ok=False)
    history = rnd.history()
    for path in rnd.parent.descriptor["code_paths"]:
        rnd.parent.path(path)
    checkpoints = [path for path in rnd.parent.descriptor["checkpoint_paths"] if path.endswith(".npz")]
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one declared CPU checkpoint")
    parent_file = rnd.parent.path(checkpoints[0])
    parent = load_model(parent_file)
    if parent.meta.get("profile") != task or parent.meta["task"] != rnd.task["task"]:
        raise ValueError("Parent and projected task differ")
    training = rows_dataset(rnd.data.training_rows(), parent)
    validation = rows_dataset(rnd.data.validation_rows(), parent)
    if set(training.ids) & set(validation.ids):
        raise ValueError("Training and validation IDs overlap")
    before, before_details = measurements(task, parent, validation)
    state = rnd.state_workspace(work / "state")
    origin, revision = reference_panel(state, validation, before_details)
    seconds = min(180., max(0., (rnd.wall_seconds or 300.) - (time.monotonic() - started) - 15.))
    deadline = time.monotonic() + seconds
    checkpoint = None
    parent_sha = digest(parent_file)
    hypothesis = "Improve the same-panel result while enforcing task-specific error controls."
    losses = []
    receipt = None
    try:
        model, losses, method = candidate_for(task, parent, training, iteration, deadline)
    except TimeoutError:
        after, detail = before, before_details
        eligible, violations = False, ["train_timeout"]
        method = {"strategy": "timeout", "adam_epochs_completed": 0}
        changes = "Training deadline elapsed before a complete candidate was available."
    else:
        after, detail = measurements(task, model, validation)
        eligible, violations = decision(task, origin, before, after, before_details, detail)
        if task == "C" and iteration == len(C_SCHEDULE):
            extended, report = end_c_candidate(
                model, detail, parent, before_details, eligible, training, validation, deadline,
            )
            method["bounded_residual"] = report
            if extended is not model:
                extended_after, extended_detail = measurements(task, extended, validation)
                use, violations_extended = decision(task, origin, before, extended_after, before_details, extended_detail)
                if use:
                    model, after, detail, eligible, violations = extended, extended_after, extended_detail, use, []
                else:
                    report.update(result="promotion_rejected", violations=violations_extended)
        method.setdefault("adam_epochs_completed", len(losses))
        changes = f"Candidate built from the declared parent using {method.get('strategy', task)}."
        location = work / "candidate"
        location.mkdir()
        checkpoint = location / "checkpoint.npz"
        model.save(checkpoint)
        write_json(location / "experiment.json", {
            "method": method, "before": before, "after": after, "pass": eligible, "violations": violations,
            "parent_sha256": parent_sha, "checkpoint_sha256": digest(checkpoint), "projection_sha256": revision,
        })
        receipt = rnd.submit_candidate(
            location, hypothesis=hypothesis, change_summary=changes, seed=parent.meta["seed"],
            configuration={"method": method, "parent_sha256": parent_sha},
            checkpoint_paths=["checkpoint.npz"], code_paths=["experiment.json"], predict_argv=PREDICT_ARGV,
        )
    vote = comparable_host_vote(task, receipt, before, after)
    main = "macro_f1" if task == "A" else "rmse"
    payload = {
        "dataset": task, "round": iteration, "parent": before, "candidate": after,
        "eligible": eligible, "violations": violations, "host_vote": vote,
        "recipe": method.get("strategy", "bounded_continuation"),
        "history": abbreviated_history(state, iteration, main),
    }
    system = (
        "Select an industrial model artifact using the measured evidence. Return only JSON with keys select "
        "and reason. select is candidate or parent. Choose candidate only when eligible is true and "
        "host_vote is not reject. Otherwise choose parent. Use only supplied measurements; never claim hidden scores."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
    answer = rnd.model(messages, max_tokens=min(160, rnd.max_output_tokens), temperature=0, enable_thinking=False)
    selection, reason = parse_choice(answer.text)
    adopt = receipt is not None and eligible and vote != "reject" and selection == "candidate"
    selected = receipt["artifact_id"] if adopt else rnd.parent.artifact_id
    if violations:
        reason += " Quality checks: " + ", ".join(violations) + "."
    audit = {
        "dataset": task, "round": iteration, "method": method, "before": before, "after": after,
        "before_details": before_details, "after_details": detail, "losses": losses,
        "eligible": eligible, "violations": violations, "host_vote": vote,
        "parent_sha256": parent_sha, "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
        "public_panel_revision": revision, "history_ids": [item["id"] for item in history],
        "select": "candidate" if adopt else "parent", "selected_artifact_id": selected,
        "model_call_id": answer.call_id, "model_messages": messages, "model_response": answer.text,
        "host_candidate_receipt": receipt, "reason": reason,
    }
    write_json(state / f"next-round-{iteration:02d}.json", audit)
    rnd.save_state(state)
    rnd.finish(
        selected_artifact_id=selected, hypothesis=hypothesis, change_summary=changes, reason=reason,
        configuration={"method": method, "guard_passed": eligible, "host_vote": vote},
        seed=parent.meta["seed"], history_ids=list(rnd.history_ids), model_call_ids=list(rnd.call_ids),
    )


def predict(sample, model_dir):
    base = Path(model_dir).resolve()
    file = Path(os.environ.get("CONTEST_ARTIFACT_PATH", str(base / "checkpoint.npz")))
    path = file.resolve()
    if file.is_symlink() or not path.is_relative_to(base):
        raise ValueError("Selected checkpoint must be inside the model directory")
    if path not in _MODELS:
        _MODELS[path] = load_model(path)
    model = _MODELS[path]
    if sample.task != model.meta["task"]:
        raise ValueError("Prediction task differs from the checkpoint")
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("Prediction features differ from the checkpoint")
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if sample.task == "regression":
        return {"value": float(output)}
    labels = model.meta["classes"]
    maximum = output.max()
    if np.count_nonzero(output == maximum) > 1:
        output[int(np.argmax(output))] = np.nextafter(maximum, 1.0)
    return [{"label": label, "score": float(value)} for label, value in zip(labels, output, strict=True)]
