"""Tests for the When2Call matched benchmark adapter and evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from eval_when2call import category_metrics
from when2call_adapter import (
    ANSWER_CATEGORIES,
    SHORT_LABELS,
    build_when2call_cases,
    request_body,
)


def _row(index: int, gold: str, tools: list[str] | None = None) -> dict:
    return {
        "uuid": f"case-{index}",
        "source": "fixture",
        "source_id": str(index),
        "question": f"question {index}",
        "correct_answer": gold,
        "answers": {category: f"response {category} {index}" for category in ANSWER_CATEGORIES},
        "tools": tools or [],
    }


class When2CallTests(unittest.TestCase):

    def test_adapter_is_balanced_for_a_smoke_limit_and_does_not_leak_gold(self):
        rows = [
            _row(i, category, ['{"name":"f","parameters":{"type":"object"}}'])
            for i, category in enumerate(ANSWER_CATEGORIES * 3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cases = build_when2call_cases(path, limit=8, seed=7)
        self.assertEqual(len(cases), 8)
        self.assertEqual({case["gold_category"] for case in cases}, set(ANSWER_CATEGORIES))
        for case in cases:
            self.assertEqual(case["candidate_labels"], list(SHORT_LABELS))
            body = request_body(case, "model")
            rendered = json.dumps(body)
            self.assertNotIn("gold_category", rendered)
            self.assertNotIn("correct_label", rendered)

    def test_category_metrics_include_official_hallucination_rate(self):
        rows = []
        predictions = {
            "direct": "direct",
            "tool_call": "tool_call",
            "request_for_info": "cannot_answer",
            "cannot_answer": "tool_call",
        }
        for category in ANSWER_CATEGORIES:
            rows.append({
                "gold_category": category,
                "predicted_category": predictions[category],
                "official_hallucination_eligible": category == "cannot_answer",
            })
        metrics = category_metrics(rows)
        self.assertEqual(metrics["official_tool_hallucination_n"], 1)
        self.assertEqual(metrics["official_tool_hallucination_rate"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["request_for_info"]["cannot_answer"], 1)


if __name__ == "__main__":
    unittest.main()
