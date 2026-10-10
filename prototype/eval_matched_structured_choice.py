"""Paired evaluation of Clef and its Qwen3.8-27B base on identical choices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval_candidate_choice import (
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    _bootstrap_metrics,
    _ci,
    _derived_seed,
    _multiclass_brier,
    _summarize,
)


def _rows(path: Path, condition: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifact = json.load(path.open())
    rows = {
        row["case_id"]: row
        for row in artifact["results"]
        if row.get("error") is None and row["condition"] == condition
    }
    if not rows:
        raise RuntimeError(f"No successful {condition!r} rows in {path}")
    return artifact, rows


def _mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant
    return min(1.0, 2.0 * tail)


def evaluate(
    clef_path: Path,
    qwen_path: Path,
    condition: str = "short",
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    clef_artifact, clef = _rows(clef_path, condition)
    qwen_artifact, qwen = _rows(qwen_path, condition)
    case_ids = sorted(set(clef) & set(qwen))
    if len(case_ids) != len(clef) or len(case_ids) != len(qwen):
        raise ValueError(
            f"Runs are not exactly paired: clef={len(clef)}, qwen={len(qwen)}, "
            f"intersection={len(case_ids)}"
        )

    clef_rows, qwen_rows = [], []
    for case_id in case_ids:
        left, right = clef[case_id], qwen[case_id]
        for field in ("correct_label", "candidate_labels", "candidate_kinds", "candidate_actions"):
            if left[field] != right[field]:
                raise ValueError(f"Pair mismatch for {case_id}: {field}")
        clef_rows.append(left)
        qwen_rows.append(right)

    clef_correct = np.asarray([row["correct"] for row in clef_rows], dtype=float)
    qwen_correct = np.asarray([row["correct"] for row in qwen_rows], dtype=float)
    accuracy_delta = clef_correct - qwen_correct
    clef_brier = np.asarray([_multiclass_brier(row) for row in clef_rows])
    qwen_brier = np.asarray([_multiclass_brier(row) for row in qwen_rows])
    p_gold_delta = np.asarray([
        left["probabilities"][left["correct_label"]]
        - right["probabilities"][right["correct_label"]]
        for left, right in zip(clef_rows, qwen_rows)
    ])
    same_choice = np.asarray([
        left["candidate_labels"].index(left["choice"])
        == right["candidate_labels"].index(right["choice"])
        for left, right in zip(clef_rows, qwen_rows)
    ])
    left_only = int(np.sum((clef_correct == 1) & (qwen_correct == 0)))
    right_only = int(np.sum((clef_correct == 0) & (qwen_correct == 1)))

    clef_summary = _summarize(clef_rows)
    qwen_summary = _summarize(qwen_rows)
    clef_summary["bootstrap_ci95"] = _bootstrap_metrics(
        clef_rows, bootstrap_reps, _derived_seed(bootstrap_seed, "clef")
    )
    qwen_summary["bootstrap_ci95"] = _bootstrap_metrics(
        qwen_rows, bootstrap_reps, _derived_seed(bootstrap_seed, "qwen")
    )
    return {
        "protocol": {
            "condition": condition,
            "n": len(case_ids),
            "bootstrap_reps": bootstrap_reps,
            "bootstrap_seed": bootstrap_seed,
            "clef_model_info": clef_artifact.get("model_info"),
            "qwen_model_info": qwen_artifact.get("model_info"),
        },
        "clef": clef_summary,
        "qwen_structured": qwen_summary,
        "paired": {
            "accuracy_difference_clef_minus_qwen": float(accuracy_delta.mean()),
            "accuracy_difference_ci95": _ci(
                accuracy_delta, bootstrap_reps,
                _derived_seed(bootstrap_seed, "accuracy_delta"),
            ),
            "multiclass_brier_difference_clef_minus_qwen": float(
                (clef_brier - qwen_brier).mean()
            ),
            "multiclass_brier_difference_ci95": _ci(
                clef_brier - qwen_brier, bootstrap_reps,
                _derived_seed(bootstrap_seed, "multiclass_brier_delta"),
            ),
            "mean_p_gold_difference_clef_minus_qwen": float(p_gold_delta.mean()),
            "mean_p_gold_difference_ci95": _ci(
                p_gold_delta, bootstrap_reps,
                _derived_seed(bootstrap_seed, "p_gold_delta"),
            ),
            "same_candidate_rate": float(same_choice.mean()),
            "clef_only_correct": left_only,
            "qwen_only_correct": right_only,
            "mcnemar_exact_p": _mcnemar(left_only, right_only),
            "mean_latency_ratio_qwen_over_clef": float(
                qwen_summary["mean_latency_s"] / clef_summary["mean_latency_s"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clef", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("--condition", default="short")
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=Path("matched_structured_choice_metrics.json"))
    args = parser.parse_args()
    result = evaluate(
        args.clef, args.qwen, args.condition, args.bootstrap_reps, args.bootstrap_seed
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

