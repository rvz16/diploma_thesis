"""Build paired BFCL candidate-choice cases for testing H1--H3.

The benchmark turns each representable BFCL ``multiple`` item into a bounded
decision.  Every case contains the oracle action plus schema-valid distractors:
an argument near-miss when possible, the lexically closest wrong tool, and the
most distant wrong tool.  The same candidates are exposed under short and long
arbitrary labels so label/tokenization sensitivity can be measured without
changing the underlying decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "bfcl_v4_6ea57973"
DEFAULT_FEATURES = ROOT / "features_bfcl_multiple.json"
DEFAULT_FEATURES_3B = ROOT / "features_bfcl_multiple_3b.json"
SHORT_LABELS = ("A", "B", "C", "D")
LONG_LABELS = (
    "candidate_option_alpha",
    "candidate_option_bravo",
    "candidate_option_charlie",
    "candidate_option_delta",
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _words(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _function_text(function: dict[str, Any]) -> str:
    params = function.get("parameters", {}).get("properties", {})
    return " ".join([
        function.get("name", ""),
        function.get("description", ""),
        *params,
        *(str(spec.get("description", "")) for spec in params.values()),
    ])


def function_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = _words(_function_text(left)), _words(_function_text(right))
    return len(a & b) / len(a | b) if a or b else 0.0


def _first_gold_value(values: Any, *, optional: bool) -> tuple[bool, Any]:
    candidates = values if isinstance(values, list) else [values]
    if optional and "" in candidates:
        return False, None
    for value in candidates:
        if value != "":
            return True, value
    return False, None


def gold_action(answer: dict[str, Any], functions: list[dict[str, Any]]) -> dict[str, Any]:
    # The constrained baseline emits one action, so use the same first-call
    # interpretation as bfcl_adapter.build_bfcl.
    call = answer["ground_truth"][0]
    name, gold_args = next(iter(call.items()))
    function = next(function for function in functions if function["name"] == name)
    required = set(function.get("parameters", {}).get("required", []))
    arguments: dict[str, Any] = {}
    for key, values in gold_args.items():
        include, value = _first_gold_value(values, optional=key not in required)
        if include:
            arguments[key] = value
    return {"name": name, "arguments": arguments}


def _type(spec: dict[str, Any]) -> str | None:
    value = spec.get("type")
    if isinstance(value, list):
        value = next((item for item in value if item != "null"), None)
    return {"number": "number", "float": "number", "integer": "integer",
            "boolean": "boolean", "string": "string", "array": "array",
            "object": "object", "dict": "object"}.get(value, value)


def _value_valid(value: Any, spec: dict[str, Any]) -> bool:
    if "enum" in spec and value not in spec["enum"]:
        return False
    kind = _type(spec)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return True


def action_is_schema_valid(action: dict[str, Any], functions: list[dict[str, Any]]) -> bool:
    function = next((item for item in functions if item["name"] == action.get("name")), None)
    arguments = action.get("arguments")
    if function is None or not isinstance(arguments, dict):
        return False
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    if not required.issubset(arguments) or any(key not in properties for key in arguments):
        return False
    return all(_value_valid(value, properties[key]) for key, value in arguments.items())


def _default_value(spec: dict[str, Any], reused: Any = None) -> Any:
    if reused is not None and _value_valid(reused, spec):
        return reused
    enum = spec.get("enum")
    if enum:
        return enum[0]
    if "default" in spec:
        return spec["default"]
    kind = _type(spec)
    if kind == "boolean":
        return False
    if kind == "integer":
        return int(spec.get("minimum", 0))
    if kind == "number":
        return float(spec.get("minimum", 0.0))
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "unknown"


def action_for_function(function: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    gold_values = gold.get("arguments", {})
    arguments = {
        key: _default_value(properties[key], gold_values.get(key))
        for key in required if key in properties
    }
    return {"name": function["name"], "arguments": arguments}


def _mutations(value: Any, spec: dict[str, Any]) -> list[Any]:
    enum = spec.get("enum")
    if enum:
        return [candidate for candidate in enum if candidate != value]
    kind = _type(spec)
    if kind == "boolean":
        return [not value]
    if kind in {"integer", "number"} and isinstance(value, (int, float)):
        return [value + 1, value - 1]
    if kind == "string" and isinstance(value, str) and not spec.get("pattern"):
        return [value + "_other", "unknown"]
    if kind == "array" and isinstance(value, list):
        return [value + ["unknown"], [] if value else ["unknown"]]
    return []


def argument_near_miss(gold: dict[str, Any], function: dict[str, Any]) -> dict[str, Any] | None:
    properties = function.get("parameters", {}).get("properties", {})
    for key, value in gold["arguments"].items():
        for candidate in _mutations(value, properties.get(key, {})):
            action = {"name": gold["name"], "arguments": dict(gold["arguments"])}
            action["arguments"][key] = candidate
            if action != gold and action_is_schema_valid(action, [function]):
                return action
    return None


def _depth(value: Any) -> int:
    if isinstance(value, dict) and value:
        return 1 + max(_depth(item) for item in value.values())
    if isinstance(value, list) and value:
        return 1 + max(_depth(item) for item in value)
    return 0


def schema_complexity(functions: list[dict[str, Any]]) -> dict[str, int]:
    properties = [
        spec
        for function in functions
        for spec in function.get("parameters", {}).get("properties", {}).values()
    ]
    required = sum(len(function.get("parameters", {}).get("required", []))
                   for function in functions)
    return {
        "n_tools": len(functions),
        "n_properties": len(properties),
        "n_required": required,
        "max_schema_depth": max((_depth(spec) for spec in properties), default=0),
        "score": len(functions) + len(properties) + required
        + max((_depth(spec) for spec in properties), default=0),
    }


def _action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def build_base_cases(data_dir: Path = DEFAULT_DATA,
                     features_path: Path = DEFAULT_FEATURES,
                     features_3b_path: Path | None = DEFAULT_FEATURES_3B,
                     seed: int = 2026) -> list[dict[str, Any]]:
    entries = {row["id"]: row for row in _json_lines(data_dir / "BFCL_v4_multiple.json")}
    answers = {row["id"]: row for row in _json_lines(
        data_dir / "possible_answer" / "BFCL_v4_multiple.json"
    )}
    with features_path.open() as stream:
        features_blob = json.load(stream)
    features = {row["id"]: row for row in features_blob["rows"]}
    feature_sets = {features_blob["model"]: features}
    if features_3b_path is not None:
        with features_3b_path.open() as stream:
            features_3b_blob = json.load(stream)
        feature_sets[features_3b_blob["model"]] = {
            row["id"]: row for row in features_3b_blob["rows"]
        }
    cases = []
    for item_id, baseline in features.items():
        entry, answer = entries[item_id], answers[item_id]
        functions = entry["function"]
        gold = gold_action(answer, functions)
        gold_function = next(function for function in functions if function["name"] == gold["name"])
        alternatives = [function for function in functions if function["name"] != gold["name"]]
        ranked = sorted(
            ((function_similarity(gold_function, function), function) for function in alternatives),
            key=lambda pair: pair[0],
        )
        candidates: list[dict[str, Any]] = [{"kind": "gold", "action": gold, "similarity": 1.0}]
        near_miss = argument_near_miss(gold, gold_function)
        if near_miss is not None:
            candidates.append({"kind": "wrong_argument", "action": near_miss, "similarity": 1.0})
        if ranked:
            similarity, function = ranked[-1]
            candidates.append({"kind": "near_wrong_tool", "action": action_for_function(function, gold),
                               "similarity": similarity})
        if len(ranked) > 1:
            similarity, function = ranked[0]
            candidates.append({"kind": "far_wrong_tool", "action": action_for_function(function, gold),
                               "similarity": similarity})

        unique: list[dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            key = _action_key(candidate["action"])
            if key not in seen and action_is_schema_valid(candidate["action"], functions):
                unique.append(candidate)
                seen.add(key)
        if len(unique) < 2:
            continue
        unique = unique[:4]
        item_seed = int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()[:16], 16)
        random.Random(item_seed).shuffle(unique)
        cases.append({
            "id": item_id,
            "user_request": entry["question"][0][0]["content"],
            "available_tools": functions,
            "candidates": unique,
            "gold_index": next(i for i, candidate in enumerate(unique) if candidate["kind"] == "gold"),
            "schema_complexity": schema_complexity(functions),
            "near_tool_similarity": max((candidate["similarity"] for candidate in unique
                                         if candidate["kind"] == "near_wrong_tool"), default=None),
            "baselines": {
                model: {
                    "model": model,
                    "correct": int(rows[item_id]["correct"]),
                    "wrong_valid": int(rows[item_id]["wrong_valid"]),
                    "gnll_smt": float(rows[item_id]["gnll_smt"]),
                    "semantic_token_confidence": float(math.exp(-rows[item_id]["gnll_smt"])),
                    "prediction": rows[item_id]["pred"],
                }
                for model, rows in feature_sets.items() if item_id in rows
            },
        })
    return cases


def encode_case(case: dict[str, Any], condition: str) -> dict[str, Any]:
    labels = SHORT_LABELS if condition == "short" else LONG_LABELS
    if condition not in {"short", "long"}:
        raise ValueError("condition must be 'short' or 'long'")
    labels = labels[:len(case["candidates"])]
    criteria = {
        label: "Candidate action: " + json.dumps(candidate["action"], sort_keys=True)
        for label, candidate in zip(labels, case["candidates"])
    }
    return {
        "case_id": case["id"],
        "condition": condition,
        "state": {
            "task": "Select the semantically correct function call for the user request.",
            "user_request": case["user_request"],
            "available_tools": case["available_tools"],
        },
        "question": {
            "type": "choice",
            "instructions": (
                "Choose exactly one candidate action. All candidates are schema-valid; "
                "select the tool and arguments that best satisfy the user request."
            ),
            "criteria": criteria,
        },
        "correct_label": labels[case["gold_index"]],
        "candidate_labels": list(labels),
        "candidates": case["candidates"],
        "schema_complexity": case["schema_complexity"],
        "near_tool_similarity": case["near_tool_similarity"],
        "baselines": case["baselines"],
    }


def build_encoded_cases(data_dir: Path = DEFAULT_DATA,
                        features_path: Path = DEFAULT_FEATURES,
                        features_3b_path: Path | None = DEFAULT_FEATURES_3B,
                        conditions: tuple[str, ...] = ("short", "long"),
                        seed: int = 2026,
                        limit: int | None = None) -> list[dict[str, Any]]:
    cases = build_base_cases(data_dir, features_path, features_3b_path, seed)
    random.Random(seed).shuffle(cases)
    if limit:
        cases = cases[:limit]
    return [encode_case(case, condition) for case in cases for condition in conditions]


def request_body(case: dict[str, Any], model: str) -> dict[str, Any]:
    # Evaluation-only fields such as correct_label and baseline never enter the
    # model request.
    return {
        "model": model,
        "state": case["state"],
        "questions": {"action": case["question"]},
    }
