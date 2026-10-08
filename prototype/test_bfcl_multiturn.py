"""Tests for the BFCL multi-turn trajectory instrumentation (no model needed)."""

import ast
import json
import sys
import unittest

sys.path.insert(0, ".")

from bfcl_multiturn import (
    DEFAULT_DATA,
    END_NAME,
    load_entries,
    load_tools,
    rollout_entry,
    trajectory_record,
)
from fc_constrained import DecodeResult, Step


def call_to_json(call, reverse_names, arg_order):
    node = ast.parse(call, mode="eval").body
    args = {name: ast.literal_eval(value)
            for name, value in zip(arg_order[node.func.id], node.args)}
    args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    return {
        "name": reverse_names[node.func.id],
        "arguments": args,
    }


class FakeDecoder:
    def __init__(self, actions):
        self.actions = iter(actions)

    def generate_schema(self, system, user, tools):
        obj = next(self.actions)
        return DecodeResult(text=json.dumps(obj), parsed=obj,
                            steps=[Step("function", obj["name"], 0.9, 0.8)])


class Instance:
    def __init__(self):
        self.calls = []


class FakeExecutor:
    def __init__(self):
        self.instance = Instance()

    def __call__(self, calls, *args, **kwargs):
        for call in calls:
            self.instance.calls.append(call)
        return [f"executed: {call}" for call in calls], {"FakeAPI": self.instance}


class MultiTurnTests(unittest.TestCase):
    def test_pinned_base_data_and_rollout_log(self):
        entry, gold = load_entries(DEFAULT_DATA, limit=1)[0]
        tools, mapping, _, arg_order = load_tools(entry, DEFAULT_DATA)
        self.assertGreater(len(tools), 0)
        reverse = {raw: namespaced for namespaced, raw in mapping.items()}
        actions = []
        for turn in gold:
            actions.extend(call_to_json(call, reverse, arg_order) for call in turn)
            actions.append({"name": END_NAME, "arguments": {}})
        executor = FakeExecutor()
        result = rollout_entry(entry, gold, FakeDecoder(actions), executor,
                               lambda *args: {"valid": True}, DEFAULT_DATA, "test_run")
        self.assertEqual(result["trajectory_success"], 1)
        self.assertEqual(result["trajectory_failure"], 0)
        self.assertEqual(len(executor.instance.calls), sum(map(len, gold)))
        self.assertTrue(all("state_before_sha256" in r for r in result["rows"]))
        self.assertFalse(any(r["wrong_valid"] for r in result["rows"]))
        record = trajectory_record(result)
        self.assertEqual(record["trajectory_failure"], 0)
        self.assertGreater(len(record["trajectory_steps"]), 0)


if __name__ == "__main__":
    unittest.main()
