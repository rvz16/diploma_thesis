"""API-Bank level-1, teacher-forced next-call evaluation for CP baselines.

Only scalar-schema dialogues are included; incompatible APIs are never silently
removed from the supplied tool inventory. Call correctness is strict reference
name/argument equality, not the original execution-based leaderboard metric.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "apibank_level1_faf75467.json"
SOURCE_REVISION = "faf754671f5b850e1b4d1ff870db37a6a2480df6"
TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def build_bundle(source_root: Path) -> dict[str, Any]:
    """Read official literal API declarations without importing API modules."""
    metadata = {}
    hashes = {}
    for path in sorted((source_root / "apis").glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            attrs = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id in (
                            "description", "input_parameters", "output_parameters"
                        ):
                            attrs[target.id] = ast.literal_eval(statement.value)
            if "input_parameters" in attrs:
                metadata[node.name] = attrs
                hashes[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dialogues = []
    for path in sorted((source_root / "lv1-lv2-samples" / "level-1-given-desc").glob("*.jsonl")):
        dialogues.append({
            "id": path.name,
            "messages": [json.loads(line) for line in path.read_text().splitlines() if line.strip()],
        })
        hashes[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_repository": "https://github.com/AlibabaResearch/DAMO-ConvAI",
        "source_revision": SOURCE_REVISION,
        "protocol": "API-Bank level-1-given-desc; teacher-forced next API call",
        "source_file_sha256": hashes,
        "api_metadata": metadata,
        "dialogues": dialogues,
    }


def _tools(names: list[str], metadata: dict[str, Any]) -> list[tuple[str, list]] | None:
    tools = []
    for name in names:
        params = []
        for key, declaration in metadata[name]["input_parameters"].items():
            if declaration["type"] not in TYPES:
                return None
            params.append((key, {"type": TYPES[declaration["type"]]}, True))
        tools.append((name, params))
    return tools


def schema_valid(parsed: Any, tools: list[tuple[str, list]]) -> bool:
    if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
        return False
    params = next((params for name, params in tools if name == parsed["name"]), None)
    args = parsed["arguments"]
    if params is None or not isinstance(args, dict) or set(args) != {key for key, _, _ in params}:
        return False
    for key, spec, _ in params:
        value, kind = args[key], spec["type"]
        valid = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }[kind]
        if not valid:
            return False
    return True


def exact_correct(parsed: Any, task: dict[str, Any]) -> bool:
    return schema_valid(parsed, task["tools"]) and parsed == task["gold"]


def build_tasks(data: Path = DEFAULT_DATA, limit: int | None = None,
                seed: int = 2026) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = json.loads(data.read_text())
    metadata = bundle["api_metadata"]
    counts = Counter()
    tasks = []
    for dialogue in bundle["dialogues"]:
        messages = dialogue["messages"]
        # Match the original level-1 given-description inventory: all APIs
        # appearing in this dialogue, without revealing their order/arguments.
        names = sorted({message["api_name"] for message in messages if message["role"] == "API"})
        tools = _tools(names, metadata)
        docs = [{"name": name, **metadata[name]} for name in names]
        for position, target in enumerate(messages):
            if target["role"] != "API":
                continue
            counts["raw_api_calls"] += 1
            if tools is None:
                counts["excluded_unsupported_dialogue_schema"] += 1
                continue
            gold = {"name": target["api_name"], "arguments": target["param_dict"]}
            if not schema_valid(gold, tools):
                counts["excluded_reference_schema_mismatch"] += 1
                continue
            tasks.append({
                "id": f"{dialogue['id']}::{position}",
                "dialogue_id": dialogue["id"],
                "position": position,
                "family": target["api_name"],
                "system": (
                    "You are a function-calling assistant. This year is 2023. "
                    "Given the available API descriptions and the conversation, predict "
                    "the next API call. Return ONLY a JSON object "
                    '{"name":"<API name>","arguments":{...}}. '
                    "Include all declared input parameters. Available APIs:\n"
                    + json.dumps(docs, ensure_ascii=False)
                ),
                "user": "Conversation before the next API call:\n" + json.dumps(
                    messages[:position], ensure_ascii=False
                ),
                "tools": tools,
                "gold": gold,
            })
    random.Random(seed).shuffle(tasks)
    counts["eligible_api_calls"] = len(tasks)
    if limit:
        tasks = tasks[:limit]
    counts["selected_api_calls"] = len(tasks)
    return tasks, {
        "source_revision": bundle["source_revision"],
        "bundle_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "counts": dict(counts),
        "dialogues_selected": len({task["dialogue_id"] for task in tasks}),
        "tool_families_selected": len({task["family"] for task in tasks}),
        "seed": seed,
        "correctness_metric": "strict reference API name and typed arguments; no tool execution",
        "context_protocol": "teacher-forced history; official level-1 per-dialogue API inventory",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.source_root:
        print(json.dumps(build_bundle(args.source_root), ensure_ascii=False, separators=(",", ":")))
    else:
        tasks, manifest = build_tasks(args.data)
        print(json.dumps(manifest, indent=2))
