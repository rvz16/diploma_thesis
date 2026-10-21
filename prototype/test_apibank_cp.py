"""Schema, history-leakage and detection-population checks for API-Bank."""

import json
from pathlib import Path
import tempfile
import unittest

from apibank_adapter import build_tasks, exact_correct, schema_valid
from eval_apibank_cp import CP_FEATURES, LOGIT_FEATURES, evaluate


class APIBankTests(unittest.TestCase):
    def test_pinned_population_has_representable_references(self):
        tasks, manifest = build_tasks()
        self.assertEqual(manifest["counts"]["raw_api_calls"], 389)
        self.assertEqual(len(tasks), 290)
        for task in tasks:
            self.assertTrue(schema_valid(task["gold"], task["tools"]))
            self.assertTrue(exact_correct(task["gold"], task))
            wrong = {"name": "UnlistedTool", "arguments": {}}
            self.assertFalse(schema_valid(wrong, task["tools"]))

    def test_prompt_contains_prior_observations_but_not_current_or_future_results(self):
        bundle = {
            "source_revision": "test",
            "api_metadata": {"F": {"description": "fixture", "input_parameters": {
                "x": {"type": "str", "description": "value"}}, "output_parameters": {}}},
            "dialogues": [{"id": "fixture", "messages": [
                {"role": "User", "text": "use the tool"},
                {"role": "API", "api_name": "F", "param_dict": {"x": "first"},
                 "result": {"output": "PRIOR_OBSERVATION"}},
                {"role": "API", "api_name": "F", "param_dict": {"x": "HELD_OUT_ARGUMENT"},
                 "result": {"output": "CURRENT_OBSERVATION"}},
                {"role": "AI", "text": "FUTURE_MESSAGE"},
            ]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(bundle))
            tasks, _ = build_tasks(path)
        target = next(task for task in tasks if task["position"] == 2)
        prompt = target["system"] + target["user"]
        self.assertIn("PRIOR_OBSERVATION", prompt)
        for hidden in ("HELD_OUT_ARGUMENT", "CURRENT_OBSERVATION", "FUTURE_MESSAGE"):
            self.assertNotIn(hidden, prompt)

    def test_invalid_generation_is_separate_from_wrong_valid(self):
        rows = []
        for index in range(30):
            row = {
                "dialogue_id": f"dialogue-{index // 2}", "family": "F",
                "schema_valid": 1, "correct": index % 2,
                "wrong_valid": 1 - index % 2, "error": None,
            }
            row.update({feature: float(index % 3) for feature in LOGIT_FEATURES + CP_FEATURES})
            row["avg_sem_tok_unc"] = float(index % 3)
            rows.append(row)
        rows.append({"schema_valid": 0, "correct": 0, "wrong_valid": 0,
                     "family": "F", "error": None})
        metrics = evaluate(rows)
        self.assertEqual(metrics["invalid"], 1)
        self.assertEqual(metrics["wrong_valid"], 15)
        self.assertEqual(metrics["schema_valid"], 30)
        self.assertEqual(metrics["cv"]["group"], "source dialogue")
        self.assertIn("logit-plus-CP", metrics["oof_detectors"])


if __name__ == "__main__":
    unittest.main()
