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
)
from eval_candidate_choice import _baseline, _label_sensitivity, _summarize


class CandidateChoiceTests(unittest.TestCase):
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

    def test_short_and_long_conditions_preserve_candidate_mapping(self):
        for case in self.cases[:25]:
            short, long = encode_case(case, "short"), encode_case(case, "long")
            self.assertEqual(short["candidates"], long["candidates"])
            short_index = short["candidate_labels"].index(short["correct_label"])
            long_index = long["candidate_labels"].index(long["correct_label"])
            self.assertEqual(short_index, long_index)

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
        self.assertEqual(_label_sensitivity(rows)["same_candidate_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
