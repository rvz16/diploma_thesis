"""BFCL multi-turn rollouts with per-action constrained-decoding uncertainty.

BFCL's multi-turn protocol was introduced in V3 and is currently distributed
in the official repository under ``BFCL_v4_multi_turn_*``.  This module keeps
the official stateful executor as the source of truth; it does *not* turn a
trajectory into independent one-turn questions.

The model emits one JSON action at a time.  ``__end__`` is an internal,
non-executable sentinel meaning that the current user turn is complete.  It is
necessary because the prototype decoder emits one structured object per model
call, whereas BFCL permits an arbitrary number of calls per turn.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from bfcl_adapter import _func_to_spec
from run_poc import features_from_steps

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "bfcl_multiturn_v4_6ea57973"
END_NAME = "__end__"
MAX_ACTIONS_PER_TURN = 8

DOC_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _canonical_call(call: str, arg_order: dict[str, list[str]] | None = None) -> str | None:
    """AST-normalize calls, including BFCL's positional-vs-keyword variants."""
    try:
        tree = ast.parse(call, mode="eval")
    except SyntaxError:
        return None
    node = tree.body
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and arg_order:
        names = arg_order.get(node.func.id, [])
        if len(node.args) <= len(names):
            node.keywords.extend(ast.keyword(arg=names[i], value=value)
                                 for i, value in enumerate(node.args))
            node.args = []
        node.keywords.sort(key=lambda kw: kw.arg or "")
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _public_state(instances: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Serializable state snapshot plus a stable digest, for audit not prompting."""
    state = {
        name: {k: v for k, v in vars(instance).items() if not k.startswith("_")}
        for name, instance in instances.items()
    }
    encoded = json.dumps(state, sort_keys=True, default=repr)
    return json.loads(encoded), hashlib.sha256(encoded.encode()).hexdigest()


def _add_official_bfcl_root(path: str | None) -> None:
    """Support either installed ``bfcl_eval`` or a checked-out official repo."""
    if path:
        root = Path(path).expanduser().resolve()
        if not (root / "bfcl_eval").is_dir():
            raise FileNotFoundError(f"--bfcl-root must contain bfcl_eval/: {root}")
        sys.path.insert(0, str(root))


def official_executor(bfcl_root: str | None):
    _add_official_bfcl_root(bfcl_root)
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BFCL's official executor is required. Install bfcl_eval or pass "
            "--bfcl-root /path/to/gorilla/berkeley-function-call-leaderboard."
        ) from exc
    return execute_multi_turn_func_call, multi_turn_checker


def load_entries(data_dir: Path, limit: int | None = None) -> list[tuple[dict, list[list[str]]]]:
    data = _json_lines(data_dir / "BFCL_v4_multi_turn_base.json")
    answers = {
        row["id"]: row["ground_truth"]
        for row in _json_lines(data_dir / "possible_answer" / "BFCL_v4_multi_turn_base.json")
    }
    pairs = [(entry, answers[entry["id"]]) for entry in data if entry["id"] in answers]
    return pairs[:limit] if limit else pairs


def load_tools(entry: dict, data_dir: Path) -> tuple[list[tuple[str, list]], dict[str, str], list[dict], dict[str, list[str]]]:
    """Return compatible constrained tools, namespace->raw map, and prompt docs.

    The official executor exposes short Python method names.  We namespace them
    for decoding to avoid collisions across BFCL backends, then translate back
    to the official method name immediately before execution.
    """
    tools, names, prompt_docs, arg_order = [], {}, [], {}
    for class_name in entry["involved_classes"]:
        doc_path = data_dir / "multi_turn_func_doc" / DOC_FILE[class_name]
        for func in _json_lines(doc_path):
            spec = _func_to_spec(func)
            if spec is None:
                continue
            raw_name, params = spec
            name = f"{class_name}.{raw_name}"
            tools.append((name, params))
            names[name] = raw_name
            arg_order[raw_name] = list(func.get("parameters", {}).get("properties", {}))
            prompt_docs.append({"name": name, "description": func.get("description", ""),
                                "parameters": func.get("parameters", {})})
    return tools, names, prompt_docs, arg_order


def _python_call(parsed: dict, name_to_raw: dict[str, str]) -> str | None:
    name = parsed.get("name") if isinstance(parsed, dict) else None
    args = parsed.get("arguments", {}) if isinstance(parsed, dict) else None
    if name not in name_to_raw or not isinstance(args, dict):
        return None
    if not all(isinstance(k, str) and k.isidentifier() for k in args):
        return None
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name_to_raw[name]}({rendered})"


def _prompt(system_docs: list[dict], history: list[dict], current: list[dict]) -> tuple[str, str]:
    system = (
        "You are a stateful tool-use agent. Return exactly one JSON object "
        '`{"name": "<tool>", "arguments": {...}}` per response. '
        f'When this user turn is complete, return `{{"name": "{END_NAME}", "arguments": {{}}}}`. '
        "Do not explain your choice. Available tools:\n" + json.dumps(system_docs)
    )
    transcript = history + [{"role": "user", "content": m["content"]} for m in current]
    user = "Conversation and executed observations:\n" + json.dumps(transcript, default=repr)
    return system, user


def _step_row(result, parsed: dict | None, *, trajectory_id: str, turn: int,
              step: int, observation: str | None, state_before: str,
              state_after: str, python_call: str | None, action_match: bool | None) -> dict:
    feats = features_from_steps(result.steps)
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "step": step,
        "pred": result.text,
        "parsed": parsed,
        "python_call": python_call,
        "observation": observation,
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "is_termination": int(parsed is not None and parsed.get("name") == END_NAME),
        "action_matches_gold": action_match,
        "valid_json": int(parsed is not None),
        "wrong_valid": int(parsed is not None and python_call is not None and action_match is False),
        # Preserve raw unmasked token-distribution summaries for HTC.  They
        # belong to each *agent action*, not to arbitrary chunks of JSON.
        "token_confidences": [s.p_chosen for s in result.steps],
        "top1_confidences": [s.p_top1 for s in result.steps],
        "topk_confidences": [s.p_top5 for s in result.steps],
        **feats,
    }


def rollout_entry(entry: dict, ground_truth: list[list[str]], fc, executor, checker,
                  data_dir: Path, run_id: str) -> dict:
    tools, name_to_raw, docs, arg_order = load_tools(entry, data_dir)
    # A trajectory is only comparable if every oracle call has a representable tool.
    raw_gold_names = {ast.parse(c, mode="eval").body.func.id
                      for turn in ground_truth for c in turn}
    if not raw_gold_names.issubset(set(name_to_raw.values())):
        return {"id": entry["id"], "skipped": "oracle uses unsupported tool schema"}

    constrained_tools = tools + [(END_NAME, [])]
    execute, instances = executor([], entry["initial_config"], entry["involved_classes"],
                                  run_id, entry["id"], is_evaL_run=False)
    del execute
    history, all_responses, rows = [], [], []
    force_terminated = False

    for turn, current_messages in enumerate(entry["question"]):
        # Keep the ordinary chat order: user request, then assistant action,
        # then the tool observation.  The full prior trajectory is passed back
        # to the model, whereas state snapshots stay audit-only.
        history.extend({"role": "user", "content": m["content"]} for m in current_messages)
        system, user = _prompt(docs, history, [])
        # A lexical match to *any* gold call is insufficient in a stateful
        # environment: ``mv(...)`` can be a gold call but still be critical if
        # emitted before the required ``cd(...)``.  We therefore use the next
        # reference action as a conservative causal label.  BFCL's final
        # checker remains the authoritative task-success label.
        gold_for_turn = [_canonical_call(call, arg_order) for call in ground_truth[turn]]
        turn_responses = []
        for step in range(MAX_ACTIONS_PER_TURN):
            before, before_hash = _public_state(instances)
            result = fc.generate_schema(system, user, constrained_tools)
            parsed = result.parsed
            if parsed and parsed.get("name") == END_NAME:
                row = _step_row(result, parsed, trajectory_id=entry["id"], turn=turn, step=step,
                                observation=None, state_before=before_hash, state_after=before_hash,
                                python_call=None, action_match=(not gold_for_turn))
                rows.append(row)
                break

            call = _python_call(parsed, name_to_raw) if parsed else None
            action_match = (_canonical_call(call, arg_order) == gold_for_turn[0]
                            if call and gold_for_turn else False)
            if call:
                observations, instances = executor([call], entry["initial_config"],
                                                   entry["involved_classes"], run_id, entry["id"],
                                                   is_evaL_run=False)
                observation = observations[0]
                turn_responses.append([call])
                history.extend([
                    {"role": "assistant", "content": parsed},
                    {"role": "tool", "content": observation},
                ])
                if action_match:
                    gold_for_turn.pop(0)
                system, user = _prompt(docs, history, [])
            else:
                observation = "No executable action: invalid or unsupported structured output."
                history.append({"role": "assistant", "content": result.text})
            _, after_hash = _public_state(instances)
            rows.append(_step_row(result, parsed, trajectory_id=entry["id"], turn=turn, step=step,
                                  observation=observation, state_before=before_hash,
                                  state_after=after_hash, python_call=call,
                                  action_match=action_match))
        else:
            force_terminated = True
        all_responses.append(turn_responses)

    evaluation = checker(all_responses, ground_truth, entry, "multi_turn_base", run_id)
    success = bool(evaluation.get("valid")) and not force_terminated
    first_wrong = next((i for i, row in enumerate(rows) if row["wrong_valid"]), None)
    for i, row in enumerate(rows):
        row["trajectory_success"] = int(success)
        row["trajectory_failure"] = int(not success)
        row["critical_step"] = int(not success and i == first_wrong)
    return {
        "id": entry["id"], "skipped": None, "trajectory_success": int(success),
        "force_terminated": force_terminated, "evaluation": evaluation,
        "rows": rows, "model_responses": all_responses,
    }


def trajectory_record(result: dict) -> dict | None:
    """Adapt one rollout to HTC's genuine action-sequence input format.

    SAUP's situation weights and UProp's alternative-decision distances require
    additional sampling; they are intentionally not fabricated from a greedy
    rollout.  The artifact therefore enables HTC immediately and leaves those
    methods explicitly unavailable until the MC sampling extension is run.
    """
    if result.get("skipped"):
        return None
    action_rows = [r for r in result["rows"] if r["python_call"] is not None]
    return {
        "trajectory_id": result["id"],
        "wrong_valid": result["trajectory_failure"],  # evaluator compatibility
        "trajectory_failure": result["trajectory_failure"],
        "trajectory_steps": [{
            "token_confidences": r["token_confidences"],
            "top1_confidences": r["top1_confidences"],
            "topk_confidences": r["topk_confidences"],
        } for r in action_rows],
    }


def run_evaluation(model: str, limit: int | None, data_dir: Path,
                   bfcl_root: str | None, bf16: bool, output: str,
                   dry_run: bool = False, offline: bool = False) -> dict | None:
    """Run one reproducible BFCL multi-turn evaluation or structural dry run."""
    import torch
    from fc_constrained import ConstrainedFC
    pairs = load_entries(data_dir, limit)
    compatible = [p for p in pairs if load_tools(p[0], data_dir)[0]]
    print(f"BFCL multi-turn base: {len(compatible)}/{len(pairs)} entries have constrained tools")
    if dry_run:
        print("first:", compatible[0][0]["id"])
        return None
    executor, checker = official_executor(bfcl_root)
    dtype = torch.bfloat16 if bf16 else torch.float32
    fc = ConstrainedFC(model, id2str_cache="id2str_bfcl_multiturn.pkl", dtype=dtype,
                       local_files_only=offline)
    run_id = "structured_cp_" + uuid.uuid4().hex
    results = [rollout_entry(entry, gold, fc, executor, checker, data_dir, run_id)
               for entry, gold in compatible]
    rows = [row for result in results for row in result.get("rows", [])]
    trajectory_records = [r for result in results if (r := trajectory_record(result))]
    payload = {
        "model": model, "dataset": "BFCL_v4_multi_turn_base",
        "protocol": "BFCL V3 multi-turn protocol, current V4 distribution",
        "official_executor_run_id": run_id, "trajectories": results, "rows": rows,
        "trajectory_records": trajectory_records,
    }
    with open(output, "w") as f:
        json.dump(payload, f, indent=2, default=repr)
    completed = [r for r in results if not r.get("skipped")]
    print(f"wrote {output}; trajectories={len(completed)} "
          f"success={sum(r['trajectory_success'] for r in completed)}/{len(completed)}")
    return payload


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--bfcl-root", help="Official BFCL package root containing bfcl_eval/.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="Use cached Hugging Face files only; never make a network request.")
    p.add_argument("--output", default="features_bfcl_multiturn_base.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.offline:
        # Transformers consults this setting for a few metadata paths in
        # addition to the explicit local_files_only argument below.
        os.environ["HF_HUB_OFFLINE"] = "1"
    return run_evaluation(args.model, args.limit, args.data_dir, args.bfcl_root,
                          args.bf16, args.output, args.dry_run, args.offline)


if __name__ == "__main__":
    main()
