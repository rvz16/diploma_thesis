"""Adapter for the official NVIDIA When2Call multiple-choice test set.

The original benchmark asks a model to choose one of four concrete responses:
answer directly, call a tool, request missing information, or state that the
request cannot be answered.  We retain the response texts and tool inventory,
but expose the decision through neutral A/B/C/D identifiers.  This gives Clef
and its Qwen3.8-27B backbone exactly the same finite output space.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


WHEN2CALL_REVISION = "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"
WHEN2CALL_URL = (
    "https://raw.githubusercontent.com/NVIDIA/When2Call/"
    f"{WHEN2CALL_REVISION}/data/test/when2call_test_mcq.jsonl"
)
WHEN2CALL_SHA256 = "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c"
ANSWER_CATEGORIES = ("direct", "tool_call", "request_for_info", "cannot_answer")
SHORT_LABELS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_when2call_data(path: Path | None = None) -> Path:
    """Return a verified local copy, downloading the pinned official file."""
    if path is None:
        path = Path(tempfile.gettempdir()) / f"when2call-{WHEN2CALL_REVISION[:12]}.jsonl"
    if path.exists() and _sha256(path) == WHEN2CALL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(WHEN2CALL_URL, temporary)
    actual = _sha256(temporary)
    if actual != WHEN2CALL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"When2Call checksum mismatch: expected {WHEN2CALL_SHA256}, got {actual}"
        )
    temporary.replace(path)
    return path


def load_when2call(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        if set(row["answers"]) != set(ANSWER_CATEGORIES):
            raise ValueError(f"Unexpected answer categories for {row.get('uuid')}")
        if row["correct_answer"] not in ANSWER_CATEGORIES:
            raise ValueError(f"Unexpected correct_answer for {row.get('uuid')}")
    return rows


def select_when2call(
    rows: list[dict[str, Any]], limit: int | None, seed: int = 2026
) -> list[dict[str, Any]]:
    """Deterministically interleave classes so every smoke run is informative."""
    groups = {category: [] for category in ANSWER_CATEGORIES}
    for row in rows:
        groups[row["correct_answer"]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict[str, Any]] = []
    while any(groups.values()):
        for category in ANSWER_CATEGORIES:
            if groups[category]:
                selected.append(groups[category].pop())
                if limit and len(selected) >= limit:
                    return selected
    return selected


def _tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(tool) if isinstance(tool, str) else tool for tool in row["tools"]]


def encode_when2call(row: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        label: row["answers"][category]
        for label, category in zip(SHORT_LABELS, ANSWER_CATEGORIES)
    }
    correct_label = SHORT_LABELS[ANSWER_CATEGORIES.index(row["correct_answer"])]
    return {
        "case_id": row["uuid"],
        "condition": "short",
        "state": {
            "task": (
                "Choose the best assistant response to the user given the available tools. "
                "Use a tool only when it can answer the request with the supplied information."
            ),
            "user_request": row["question"],
            "available_tools": _tools(row),
        },
        "question": {
            "type": "choice",
            "instructions": (
                "Choose exactly one response. It may answer directly, call an applicable "
                "tool, ask for required missing information, or explain that the request "
                "cannot be answered with the available tools."
            ),
            "criteria": criteria,
        },
        "correct_label": correct_label,
        "candidate_labels": list(SHORT_LABELS),
        "candidate_kinds": list(ANSWER_CATEGORIES),
        "candidate_actions": [row["answers"][category] for category in ANSWER_CATEGORIES],
        "gold_category": row["correct_answer"],
        "source": row["source"],
        "source_id": row["source_id"],
        "tool_count": len(row["tools"]),
        "official_hallucination_eligible": (
            row["correct_answer"] == "cannot_answer" and not row["tools"]
        ),
    }


def build_when2call_cases(
    data_path: Path, limit: int | None = None, seed: int = 2026
) -> list[dict[str, Any]]:
    rows = select_when2call(load_when2call(data_path), limit, seed)
    return [encode_when2call(row) for row in rows]


def request_body(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "state": case["state"],
        "questions": {"action": case["question"]},
    }
