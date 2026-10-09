"""Leakage and selection tests for the Jev BFCL gate runner."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from run_jev_bfcl_gate import _jsonable, action_key, build_state, select_actions
from run_clearml_open_decision_gate import request_body


class JevGateTests(unittest.TestCase):
    def test_selection_is_prefix_stable(self):
        rows = [{"trajectory_id": "t", "turn": 0, "step": i,
                 "python_call": "x()"} for i in range(10)]
        self.assertEqual(select_actions(rows, 4, 2026), select_actions(rows, 8, 2026)[:4])

    def test_state_excludes_current_observation_and_labels(self):
        rows = [
            {"trajectory_id": "t", "turn": 0, "step": 0, "python_call": "a()",
             "parsed": {"name": "A", "arguments": {}}, "observation": "prior obs",
             "wrong_valid": 0},
            {"trajectory_id": "t", "turn": 0, "step": 1, "python_call": "b()",
             "parsed": {"name": "B", "arguments": {}}, "observation": "SECRET future",
             "wrong_valid": 1},
        ]
        entry = {"question": [[{"content": "do the task"}]], "involved_classes": []}
        state = build_state(entry, rows, rows[1], Path("."))
        rendered = repr(state)
        self.assertIn("prior obs", rendered)
        self.assertNotIn("SECRET future", rendered)
        self.assertNotIn("wrong_valid", rendered)
        self.assertEqual(state["proposed_action"]["name"], "B")

    def test_action_key_is_stable(self):
        self.assertEqual(action_key({"trajectory_id": "x", "turn": 2, "step": 3}), "x:2:3")

    def test_sdk_metadata_is_jsonable(self):
        class Usage:
            def model_dump(self, mode):
                self.mode = mode
                return {"input_tokens": 12}

        self.assertEqual(_jsonable(Usage()), {"input_tokens": 12})

    def test_open_model_request_uses_same_bounded_gate(self):
        state = {"proposed_action": {"name": "A", "arguments": {}}}
        body = request_body(state, "clef")
        self.assertIs(body["state"], state)
        self.assertEqual(body["questions"]["gate"]["type"], "choice")
        self.assertEqual(set(body["questions"]["gate"]["criteria"]), {"execute", "review"})


if __name__ == "__main__":
    unittest.main()
