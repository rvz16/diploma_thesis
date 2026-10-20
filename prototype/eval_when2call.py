"""Paired evaluation for matched Clef/Qwen runs on When2Call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_matched_structured_choice import evaluate as evaluate_matched
from when2call_adapter import ANSWER_CATEGORIES


def _successful(path: Path) -> list[dict[str, Any]]:
    artifact = json.load(path.open())
    return [row for row in artifact["results"] if row.get("error") is None]


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        gold: {predicted: 0 for predicted in ANSWER_CATEGORIES}
        for gold in ANSWER_CATEGORIES
    }
    for row in rows:
        confusion[row["gold_category"]][row["predicted_category"]] += 1
    per_category = {}
    f1_values = []
    for category in ANSWER_CATEGORIES:
        true_positive = confusion[category][category]
        false_negative = sum(confusion[category].values()) - true_positive
        false_positive = sum(confusion[gold][category] for gold in ANSWER_CATEGORIES) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = true_positive + false_negative
        per_category[category] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        if support:
            f1_values.append(f1)
    eligible = [row for row in rows if row["official_hallucination_eligible"]]
    hallucination_rate = float(np.mean([
        row["predicted_category"] == "tool_call" for row in eligible
    ])) if eligible else 0.0
    return {
        "macro_f1_present_classes": float(np.mean(f1_values)),
        "per_category": per_category,
        "confusion_matrix": confusion,
        "official_tool_hallucination_n": len(eligible),
        "official_tool_hallucination_rate": hallucination_rate,
    }


def evaluate(clef: Path, qwen: Path, bootstrap_reps: int = 10_000,
             bootstrap_seed: int = 2026) -> dict[str, Any]:
    result = evaluate_matched(
        clef, qwen, condition="short", bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    result["clef"]["when2call"] = category_metrics(_successful(clef))
    result["qwen_structured"]["when2call"] = category_metrics(_successful(qwen))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clef", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("when2call_matched_metrics.json"))
    args = parser.parse_args()
    result = evaluate(args.clef, args.qwen, args.bootstrap_reps, args.bootstrap_seed)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
