"""Run Jev as a pre-execution gate over recorded BFCL multi-turn actions.

Jev is a bounded decision model, not an arbitrary argument generator.  This
runner therefore evaluates the fair use case suggested by the TypeSafe/Failproof
comparison: given the conversation, prior observations, available tools, and a
proposed structured call, choose between ``execute`` and ``review`` and expose
the native answer distribution.

The current action's observation and every BFCL gold label are deliberately
excluded from the Jev state.  Labels are joined only later by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "bfcl_multiturn_base_qwen3_14b_smoke.json"
DEFAULT_DATA = ROOT / "data" / "bfcl_multiturn_v4_6ea57973"
DEFAULT_OUTPUT = ROOT / "results" / "bfcl_multiturn_base_qwen3_14b_jev_gate.json"

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

GATE_INSTRUCTIONS = (
    "Decide whether the proposed tool action should be executed now to correctly "
    "advance the user's request. Use only the supplied state. Select review if "
    "the action is semantically incorrect, unsupported, unnecessary, duplicated, "
    "out of order, has wrong arguments, or should not be executed yet."
)
GATE_CRITERIA = {
    "execute": "The proposed action is the correct and appropriate next action now.",
    "review": "Do not execute this action; it needs correction, clarification, or review.",
}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_entries(data_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _json_lines(data_dir / "BFCL_v4_multi_turn_base.json")}


def load_tool_docs(entry: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for class_name in entry["involved_classes"]:
        for tool in _json_lines(data_dir / "multi_turn_func_doc" / DOC_FILE[class_name]):
            docs.append({
                "name": f"{class_name}.{tool['name']}",
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            })
    return docs


def action_key(row: dict[str, Any]) -> str:
    return f"{row['trajectory_id']}:{row['turn']}:{row['step']}"


def select_actions(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Deterministic random prefix; increasing limit preserves prior selections."""
    actions = [row for row in rows if row.get("python_call") is not None]
    random.Random(seed).shuffle(actions)
    return actions if limit is None else actions[:limit]


def build_state(entry: dict[str, Any], trajectory_rows: list[dict[str, Any]],
                current: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """Build leakage-free state available immediately before current executes."""
    current_position = next(i for i, row in enumerate(trajectory_rows)
                            if action_key(row) == action_key(current))
    prior = trajectory_rows[:current_position]
    user_turns = []
    for turn, messages in enumerate(entry["question"][: current["turn"] + 1]):
        user_turns.append({
            "turn": turn,
            "messages": [m.get("content", "") for m in messages],
        })
    history = [{
        "turn": row["turn"],
        "step": row["step"],
        "action": row.get("parsed"),
        "observation": row.get("observation"),
    } for row in prior if row.get("python_call") is not None]
    return {
        "task": "BFCL stateful tool-use decision",
        "user_turns": user_turns,
        "prior_executed_actions": history,
        "available_tools": load_tool_docs(entry, data_dir),
        "proposed_action": current["parsed"],
    }


def _load_env_key(name: str, env_file: Path = Path(".env")) -> None:
    if os.environ.get(name) or not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.removeprefix("export ").strip() == name:
            os.environ[name] = value.strip().strip('"').strip("'")
            return


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    """Convert optional SDK metadata to plain JSON without depending on its type."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--limit", type=int, default=40,
                   help="Deterministic random-prefix smoke size; omit with 0 for all actions")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--provider", choices=("openrouter", "typesafe"),
                   default="openrouter")
    p.add_argument("--model", default=None,
                   help="Defaults to the provider's official Jev latest alias")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    blob = json.load(args.input.open())
    entries = load_entries(args.data_dir)
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in blob["rows"]:
        by_trajectory.setdefault(row["trajectory_id"], []).append(row)
    for rows in by_trajectory.values():
        rows.sort(key=lambda r: (r["turn"], r["step"]))
    selected = select_actions(blob["rows"], args.limit or None, args.seed)

    if args.dry_run:
        first = selected[0]
        state = build_state(entries[first["trajectory_id"]],
                            by_trajectory[first["trajectory_id"]], first, args.data_dir)
        print(f"selected={len(selected)} first={action_key(first)}")
        print(f"state_chars={len(json.dumps(state))} tools={len(state['available_tools'])}")
        print(json.dumps(state, indent=2)[:4000])
        return

    if args.provider == "openrouter":
        key_env = "OPENROUTER_API_KEY"
        base_url = "https://openrouter.ai/api"
        model = args.model or "~typesafe/jev-latest"
    else:
        key_env = "TYPESAFE_API_KEY"
        base_url = "https://api.typesafe.ai"
        model = args.model or "jev-latest"
    _load_env_key(key_env)
    if not os.environ.get(key_env):
        raise RuntimeError(f"{key_env} is required (environment or .env)")
    try:
        from typesafe_sdk import Choice, TypeSafeClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install typesafe-sdk==0.7.2") from exc

    payload: dict[str, Any] = {
        "method": "Jev Choice pre-execution gate",
        "model": model,
        "provider": args.provider,
        "base_url": base_url,
        "source_artifact": str(args.input),
        "selection": {"seed": args.seed, "limit": args.limit or None,
                      "population_actions": sum(r.get("python_call") is not None
                                                for r in blob["rows"])},
        "gate": {"instructions": GATE_INSTRUCTIONS, "criteria": GATE_CRITERIA},
        "results": [],
    }
    if args.output.exists():
        old = json.load(args.output.open())
        if (old.get("model") == model and old.get("provider") == args.provider
                and old.get("selection") == payload["selection"]):
            payload = old
    # Provider/transport failures remain in the audit trail but must be retried
    # when a later run has working credentials or sufficient credit.
    completed = {row["action_key"] for row in payload["results"]
                 if row.get("error") is None}

    with TypeSafeClient(
        api_key=os.environ[key_env],
        base_url=base_url,
        model=model,
    ) as client:
        for index, row in enumerate(selected, 1):
            key = action_key(row)
            if key in completed:
                continue
            state = build_state(entries[row["trajectory_id"]],
                                by_trajectory[row["trajectory_id"]], row, args.data_dir)
            started = time.perf_counter()
            record: dict[str, Any] = {
                "action_key": key,
                "trajectory_id": row["trajectory_id"],
                "turn": row["turn"],
                "step": row["step"],
                "state_chars": len(json.dumps(state)),
                "prior_action_count": len(state["prior_executed_actions"]),
                "tool_count": len(state["available_tools"]),
            }
            try:
                response = client.system_one(
                    state=state,
                    questions={"gate": Choice(
                        instructions=GATE_INSTRUCTIONS,
                        criteria=GATE_CRITERIA,
                    )},
                )
                answer = response.choices["gate"]
                record.update({
                    "choice": answer.choice,
                    "confidence": float(answer.confidence),
                    "probabilities": {k: float(v) for k, v in answer.probabilities.items()},
                    "request_id": getattr(response, "request_id", None),
                    "model_version": getattr(response, "model", None),
                    "usage": _jsonable(getattr(response, "usage", None)),
                    "latency_s": time.perf_counter() - started,
                    "error": None,
                })
            except Exception as exc:  # checkpoint transport/provider failures for audit
                record.update({
                    "latency_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            payload["results"].append(record)
            _checkpoint(args.output, payload)
            status = record.get("choice", "ERROR")
            print(f"[{index}/{len(selected)}] {key} {status} {record['latency_s']:.2f}s")

    ok = sum(r.get("error") is None for r in payload["results"])
    print(f"wrote {args.output}; successful={ok}/{len(payload['results'])}")


if __name__ == "__main__":
    main()
