"""Structural and leakage tests for the BFCL candidate-choice benchmark."""

from __future__ import annotations

import json
import sys
import unittest

sys.path.insert(0, ".")

from bfcl_candidate_choice import (
    action_is_schema_valid,
    build_base_cases,
    encode_case,
    request_body,
    semantic_labels,
)
from qwen_structured_choice import single_token_choice_branch
from eval_candidate_choice import (
    _baseline,
    _bootstrap_metrics,
    _label_sensitivity,
    _semantic_vs_arbitrary,
    _summarize,
)


class CandidateChoiceTests(unittest.TestCase):

    def test_qwen_structured_choice_requires_one_token_branch(self):
        class FakeTokenizer:
            outputs = {
                '{"action":"A"}': [10, 21, 30],
                '{"action":"B"}': [10, 22, 30],
                '{"action":"C"}': [10, 23, 30],
                '{"action":"D"}': [10, 24, 30],
            }

            def encode(self, text, add_special_tokens=False):
                self.assert_false = add_special_tokens
                return self.outputs[text]

        prefix, branch = single_token_choice_branch(FakeTokenizer(), ["A", "B", "C", "D"])
        self.assertEqual(prefix, [10])
        self.assertEqual(branch, {"A": 21, "B": 22, "C": 23, "D": 24})

    @classmethod
    def setUpClass(cls):
        cls.cases = build_base_cases()

    def test_builds_large_paired_population(self):
        self.assertGreaterEqual(len(self.cases), 150)

    def test_every_case_has_one_gold_and_schema_valid_unique_candidates(self):
        for case in self.cases:
            self.assertEqual(sum(c["kind"] == "gold" for c in case["candidates"]), 1)
            rendered = [json.dumps(c["action"], sort_keys=True) for c in case["candidates"]]
            self.assertEqual(len(rendered), len(set(rendered)))
            self.assertGreaterEqual(len(rendered), 2)
            for candidate in case["candidates"]:
                self.assertTrue(action_is_schema_valid(candidate["action"], case["available_tools"]))

    def test_label_conditions_preserve_candidate_mapping(self):
        for case in self.cases[:25]:
            encoded = [encode_case(case, condition)
                       for condition in ("short", "long", "semantic")]
            self.assertTrue(all(row["candidates"] == encoded[0]["candidates"]
                                for row in encoded[1:]))
            gold_indices = [row["candidate_labels"].index(row["correct_label"])
                            for row in encoded]
            self.assertEqual(len(set(gold_indices)), 1)

    def test_semantic_labels_are_unique_and_gold_agnostic(self):
        for case in self.cases[:25]:
            labels = semantic_labels(case["candidates"])
            self.assertEqual(len(labels), len(set(labels)))
            for label, candidate in zip(labels, case["candidates"]):
                self.assertIn(candidate["action"]["name"], label)
                self.assertNotIn(candidate["kind"], label)

    def test_request_excludes_gold_and_baseline_fields(self):
        encoded = encode_case(self.cases[0], "short")
        request = request_body(encoded, "clef")
        rendered = json.dumps(request)
        self.assertNotIn("correct_label", rendered)
        self.assertNotIn("wrong_valid", rendered)
        self.assertNotIn("gnll", rendered)
        self.assertEqual(set(request["questions"]), {"action"})

    def test_evaluator_uses_native_distribution_and_paired_labels(self):
        baseline = {
            "Qwen/test": {
                "correct": 0,
                "semantic_token_confidence": 0.8,
            }
        }
        rows = [{
            "case_id": "x",
            "condition": condition,
            "correct_label": labels[0],
            "candidate_labels": labels,
            "choice": labels[0],
            "correct": 1,
            "probabilities": {labels[0]: 0.75, labels[1]: 0.25},
            "latency_s": 1.0,
            "baselines": baseline,
        } for condition, labels in (
            ("short", ["A", "B"]),
            ("long", ["candidate_option_alpha", "candidate_option_bravo"]),
        )]
        summary = _summarize(rows)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertAlmostEqual(summary["brier"], 0.0625)
        self.assertEqual(_baseline(rows, "Qwen/test")["accuracy"], 0.0)
        sensitivity = _label_sensitivity(rows)
        self.assertEqual(sensitivity["same_candidate_rate"], 1.0)
        self.assertEqual(sensitivity["mcnemar_exact_p"], 1.0)
        intervals = _bootstrap_metrics(rows, reps=100, seed=2026)
        self.assertEqual(intervals["accuracy"]["low"], 1.0)
        self.assertEqual(intervals["accuracy"]["high"], 1.0)

    def test_semantic_vs_arbitrary_is_task_paired(self):
        rows = []
        for case_id, outcomes in (("x", (1, 0, 1)), ("y", (0, 0, 1))):
            for condition, correct in zip(("short", "long", "semantic"), outcomes):
                labels = [f"{condition}_0", f"{condition}_1"]
                rows.append({
                    "case_id": case_id,
                    "condition": condition,
                    "correct_label": labels[0],
                    "candidate_labels": labels,
                    "choice": labels[0] if correct else labels[1],
                    "correct": correct,
                    "probabilities": {labels[0]: 0.75, labels[1]: 0.25},
                })
        comparison = _semantic_vs_arbitrary(rows, reps=100, seed=2026)
        self.assertEqual(comparison["n"], 2)
        self.assertEqual(comparison["semantic_accuracy"], 1.0)
        self.assertEqual(comparison["mean_arbitrary_accuracy"], 0.25)
        self.assertEqual(comparison["semantic_minus_mean_arbitrary_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
