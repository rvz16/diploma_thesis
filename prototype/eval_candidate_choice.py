"""Evaluate BFCL candidate-selection runs against constrained AR baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _ece(correct: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = ((confidence >= edges[index])
                & (confidence <= edges[index + 1] if index == bins - 1
                   else confidence < edges[index + 1]))
        if mask.any():
            total += mask.mean() * abs(float(confidence[mask].mean())
                                       - float(correct[mask].mean()))
    return float(total)


def _adaptive_ece(correct: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    groups = np.array_split(np.argsort(confidence), min(bins, len(confidence)))
    return float(sum(
        len(group) / len(confidence)
        * abs(float(confidence[group].mean()) - float(correct[group].mean()))
        for group in groups if len(group)
    ))


def _binary_metrics(correct: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    clipped = np.clip(confidence, 1e-8, 1 - 1e-8)
    return {
        "accuracy": float(correct.mean()),
        "brier": float(np.mean((confidence - correct) ** 2)),
        "ece_10": _ece(correct, confidence),
        "adaptive_ece_10": _adaptive_ece(correct, confidence),
        "log_loss": float(-np.mean(correct * np.log(clipped)
                                   + (1 - correct) * np.log(1 - clipped))),
    }


def _multiclass_brier(row: dict[str, Any]) -> float:
    labels = row["candidate_labels"]
    return float(sum(
        (float(row["probabilities"].get(label, 0.0))
         - float(label == row["correct_label"])) ** 2
        for label in labels
    ))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    correct = np.asarray([row["correct"] for row in rows], dtype=float)
    confidence = np.asarray([
        max(float(value) for value in row["probabilities"].values()) for row in rows
    ])
    result = _binary_metrics(correct, confidence)
    result.update({
        "n": len(rows),
        "multiclass_brier": float(np.mean([_multiclass_brier(row) for row in rows])),
        "mean_p_gold": float(np.mean([
            row["probabilities"][row["correct_label"]] for row in rows
        ])),
        "mean_latency_s": float(np.mean([row["latency_s"] for row in rows])),
    })
    return result


def _baseline(rows: list[dict[str, Any]], model: str) -> dict[str, float]:
    unique = {row["case_id"]: row["baselines"][model] for row in rows}
    correct = np.asarray([row["correct"] for row in unique.values()], dtype=float)
    confidence = np.asarray([row["semantic_token_confidence"] for row in unique.values()], dtype=float)
    result = _binary_metrics(correct, confidence)
    result["n"] = len(unique)
    return result


def _split_advantage(rows: list[dict[str, Any]], field: str, model: str) -> dict[str, float]:
    values = np.asarray([
        row["schema_complexity"]["score"] if field == "complexity"
        else row["near_tool_similarity"] if row["near_tool_similarity"] is not None else 0.0
        for row in rows
    ])
    median = float(np.median(values))
    low = values <= median
    high = values > median
    decision = np.asarray([row["correct"] for row in rows], dtype=float)
    baseline = np.asarray([row["baselines"][model]["correct"] for row in rows], dtype=float)
    def advantage(mask: np.ndarray) -> float:
        return float((decision[mask] - baseline[mask]).mean()) if mask.any() else float("nan")
    low_adv, high_adv = advantage(low), advantage(high)
    return {
        "median": median,
        "n_low": int(low.sum()),
        "n_high": int(high.sum()),
        "advantage_low": low_adv,
        "advantage_high": high_adv,
        "interaction_high_minus_low": high_adv - low_adv,
    }


def _label_sensitivity(rows: list[dict[str, Any]]) -> dict[str, float]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[row["case_id"]][row["condition"]] = row
    complete = [pair for pair in paired.values() if {"short", "long"}.issubset(pair)]
    if not complete:
        return {"n": 0}
    same_candidate, flips, p_differences = [], [], []
    for pair in complete:
        short, long = pair["short"], pair["long"]
        short_index = short["candidate_labels"].index(short["choice"])
        long_index = long["candidate_labels"].index(long["choice"])
        same_candidate.append(short_index == long_index)
        flips.append(short["correct"] != long["correct"])
        p_differences.append(abs(
            short["probabilities"][short["correct_label"]]
            - long["probabilities"][long["correct_label"]]
        ))
    return {
        "n": len(complete),
        "same_candidate_rate": float(np.mean(same_candidate)),
        "correctness_flip_rate": float(np.mean(flips)),
        "mean_abs_p_gold_shift": float(np.mean(p_differences)),
    }


def evaluate(path: Path) -> dict[str, Any]:
    artifact = json.load(path.open())
    rows = [row for row in artifact["results"] if row.get("error") is None]
    if not rows:
        raise RuntimeError("No successful candidate-choice responses")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    conditions = {}
    for condition, condition_rows in sorted(grouped.items()):
        baseline_models = sorted(set.intersection(*(
            set(row["baselines"]) for row in condition_rows
        )))
        decision_summary = _summarize(condition_rows)
        baseline_summaries = {
            model: _baseline(condition_rows, model) for model in baseline_models
        }
        conditions[condition] = {
            "decision_model": decision_summary,
            "constrained_ar": baseline_summaries,
            "h1_accuracy_advantage": {
                model: decision_summary["accuracy"] - summary["accuracy"]
                for model, summary in baseline_summaries.items()
            },
            "h2_schema_complexity": {
                model: _split_advantage(condition_rows, "complexity", model)
                for model in baseline_models
            },
            "h2_candidate_similarity": {
                model: _split_advantage(condition_rows, "similarity", model)
                for model in baseline_models
            },
        }
    return {
        "backend": artifact["backend"],
        "model_info": artifact["model_info"],
        "successful_requests": len(rows),
        "conditions": conditions,
        "h2_label_sensitivity": _label_sensitivity(rows),
    }


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.artifact)
    print(f"backend={result['backend']} successful_requests={result['successful_requests']}")
    print("condition  method          n   accuracy  brier  ECE10  adaptiveECE  logloss")
    for condition, block in result["conditions"].items():
        metrics = block["decision_model"]
        print(f"{condition:9s} {'decision_model':15s} {metrics['n']:3d} "
              f"{_fmt(metrics['accuracy']):>8s} {_fmt(metrics['brier']):>6s} "
              f"{_fmt(metrics['ece_10']):>6s} {_fmt(metrics['adaptive_ece_10']):>11s} "
              f"{_fmt(metrics['log_loss']):>7s}")
        for model, baseline in block["constrained_ar"].items():
            short_model = model.replace("Qwen/Qwen2.5-", "AR-").replace("-Instruct", "")
            print(f"{condition:9s} {short_model:15s} {baseline['n']:3d} "
                  f"{_fmt(baseline['accuracy']):>8s} {_fmt(baseline['brier']):>6s} "
                  f"{_fmt(baseline['ece_10']):>6s} {_fmt(baseline['adaptive_ece_10']):>11s} "
                  f"{_fmt(baseline['log_loss']):>7s}")
            print(f"  H1 advantage vs {short_model}: "
                  f"{_fmt(block['h1_accuracy_advantage'][model])}")
            for name in ("h2_schema_complexity", "h2_candidate_similarity"):
                interaction = block[name][model]
                print(f"  {name} vs {short_model}: low={_fmt(interaction['advantage_low'])} "
                      f"high={_fmt(interaction['advantage_high'])} "
                      f"interaction={_fmt(interaction['interaction_high_minus_low'])}")
    sensitivity = result["h2_label_sensitivity"]
    print("label sensitivity:", json.dumps(sensitivity, sort_keys=True))
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
