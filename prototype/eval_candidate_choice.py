"""Evaluate BFCL candidate-selection runs against constrained AR baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BOOTSTRAP_REPS = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026


def _derived_seed(seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _ci(values: np.ndarray, reps: int, seed: int) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"low": float("nan"), "high": float("nan"), "reps": reps,
                "method": "task_percentile"}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "reps": reps,
            "method": "task_percentile"}


def _bootstrap_metrics(rows: list[dict[str, Any]], reps: int, seed: int) -> dict[str, Any]:
    correct = np.asarray([row["correct"] for row in rows], dtype=float)
    confidence = np.asarray([
        max(float(value) for value in row["probabilities"].values()) for row in rows
    ])
    clipped = np.clip(confidence, 1e-8, 1 - 1e-8)
    per_row = {
        "accuracy": correct,
        "brier": (confidence - correct) ** 2,
        "log_loss": -(correct * np.log(clipped) + (1 - correct) * np.log(1 - clipped)),
        "multiclass_brier": np.asarray([_multiclass_brier(row) for row in rows]),
        "mean_p_gold": np.asarray([
            row["probabilities"][row["correct_label"]] for row in rows
        ]),
    }
    result = {
        metric: _ci(values, reps, _derived_seed(seed, metric))
        for metric, values in per_row.items()
    }

    rng = np.random.default_rng(_derived_seed(seed, "ece_10"))
    indices = rng.integers(0, len(rows), size=(reps, len(rows)))
    boot_correct, boot_confidence = correct[indices], confidence[indices]
    ece = np.zeros(reps)
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = ((boot_confidence >= edges[index])
                & (boot_confidence <= edges[index + 1] if index == 9
                   else boot_confidence < edges[index + 1]))
        count = mask.sum(axis=1)
        nonempty = count > 0
        conf_sum = (boot_confidence * mask).sum(axis=1)
        correct_sum = (boot_correct * mask).sum(axis=1)
        ece[nonempty] += (count[nonempty] / len(rows)
                          * np.abs(conf_sum[nonempty] / count[nonempty]
                                   - correct_sum[nonempty] / count[nonempty]))
    low, high = np.quantile(ece, [0.025, 0.975])
    result["ece_10"] = {"low": float(low), "high": float(high), "reps": reps,
                        "method": "task_percentile"}
    return result


def _bootstrap_baseline(rows: list[dict[str, Any]], model: str, reps: int,
                        seed: int) -> dict[str, Any]:
    unique = {row["case_id"]: row["baselines"][model] for row in rows}
    correct = np.asarray([row["correct"] for row in unique.values()], dtype=float)
    confidence = np.asarray([row["semantic_token_confidence"] for row in unique.values()])
    clipped = np.clip(confidence, 1e-8, 1 - 1e-8)
    values = {
        "accuracy": correct,
        "brier": (confidence - correct) ** 2,
        "log_loss": -(correct * np.log(clipped) + (1 - correct) * np.log(1 - clipped)),
    }
    return {
        metric: _ci(array, reps, _derived_seed(seed, f"{model}:{metric}"))
        for metric, array in values.items()
    }


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


def _split_advantage(rows: list[dict[str, Any]], field: str, model: str,
                     reps: int = DEFAULT_BOOTSTRAP_REPS,
                     seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
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
    delta = decision - baseline
    rng = np.random.default_rng(_derived_seed(seed, f"{field}:{model}:interaction"))
    low_values, high_values = delta[low], delta[high]
    low_means = low_values[rng.integers(0, len(low_values),
                                        size=(reps, len(low_values)))].mean(axis=1)
    high_means = high_values[rng.integers(0, len(high_values),
                                          size=(reps, len(high_values)))].mean(axis=1)
    interaction_ci = np.quantile(high_means - low_means, [0.025, 0.975])
    return {
        "median": median,
        "n_low": int(low.sum()),
        "n_high": int(high.sum()),
        "advantage_low": low_adv,
        "advantage_high": high_adv,
        "interaction_high_minus_low": high_adv - low_adv,
        "advantage_low_ci95": _ci(low_values, reps,
                                   _derived_seed(seed, f"{field}:{model}:low")),
        "advantage_high_ci95": _ci(high_values, reps,
                                    _derived_seed(seed, f"{field}:{model}:high")),
        "interaction_ci95": {
            "low": float(interaction_ci[0]), "high": float(interaction_ci[1]),
            "reps": reps, "method": "stratified_task_percentile",
        },
    }


def _pair_comparison(rows: list[dict[str, Any]], left_condition: str,
                     right_condition: str, reps: int = DEFAULT_BOOTSTRAP_REPS,
                     seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[row["case_id"]][row["condition"]] = row
    complete = [pair for pair in paired.values()
                if {left_condition, right_condition}.issubset(pair)]
    if not complete:
        return {"n": 0}
    same_candidate, flips, p_differences = [], [], []
    left_only_correct = 0
    right_only_correct = 0
    accuracy_differences = []
    for pair in complete:
        left, right = pair[left_condition], pair[right_condition]
        left_index = left["candidate_labels"].index(left["choice"])
        right_index = right["candidate_labels"].index(right["choice"])
        same_candidate.append(left_index == right_index)
        flips.append(left["correct"] != right["correct"])
        left_only_correct += int(left["correct"] and not right["correct"])
        right_only_correct += int(right["correct"] and not left["correct"])
        accuracy_differences.append(left["correct"] - right["correct"])
        p_differences.append(abs(
            left["probabilities"][left["correct_label"]]
            - right["probabilities"][right["correct_label"]]
        ))
    discordant = left_only_correct + right_only_correct
    if discordant:
        smaller = min(left_only_correct, right_only_correct)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    return {
        "n": len(complete),
        "same_candidate_rate": float(np.mean(same_candidate)),
        "correctness_flip_rate": float(np.mean(flips)),
        "mean_abs_p_gold_shift": float(np.mean(p_differences)),
        "left_condition": left_condition,
        "right_condition": right_condition,
        "left_accuracy": float(np.mean([pair[left_condition]["correct"] for pair in complete])),
        "right_accuracy": float(np.mean([pair[right_condition]["correct"] for pair in complete])),
        "accuracy_difference_left_minus_right": float(np.mean(accuracy_differences)),
        "accuracy_difference_ci95": _ci(
            np.asarray(accuracy_differences), reps,
            _derived_seed(seed, f"{left_condition}:{right_condition}:accuracy"),
        ),
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "mcnemar_exact_p": mcnemar_p,
    }


def _label_sensitivity(rows: list[dict[str, Any]], reps: int = DEFAULT_BOOTSTRAP_REPS,
                       seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    return _pair_comparison(rows, "short", "long", reps, seed)


def _semantic_vs_arbitrary(rows: list[dict[str, Any]], reps: int,
                           seed: int) -> dict[str, Any]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[row["case_id"]][row["condition"]] = row
    complete = [pair for pair in paired.values()
                if {"short", "long", "semantic"}.issubset(pair)]
    if not complete:
        return {"n": 0}
    differences = np.asarray([
        pair["semantic"]["correct"]
        - 0.5 * (pair["short"]["correct"] + pair["long"]["correct"])
        for pair in complete
    ])
    return {
        "n": len(complete),
        "semantic_accuracy": float(np.mean([pair["semantic"]["correct"] for pair in complete])),
        "mean_arbitrary_accuracy": float(np.mean([
            0.5 * (pair["short"]["correct"] + pair["long"]["correct"])
            for pair in complete
        ])),
        "semantic_minus_mean_arbitrary_accuracy": float(differences.mean()),
        "accuracy_difference_ci95": _ci(
            differences, reps, _derived_seed(seed, "semantic:mean_arbitrary:accuracy")
        ),
    }


def evaluate(path: Path, bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
             bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
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
        condition_seed = _derived_seed(bootstrap_seed, condition)
        decision_summary["bootstrap_ci95"] = _bootstrap_metrics(
            condition_rows, bootstrap_reps, condition_seed
        )
        baseline_summaries = {
            model: _baseline(condition_rows, model) for model in baseline_models
        }
        for model, summary in baseline_summaries.items():
            summary["bootstrap_ci95"] = _bootstrap_baseline(
                condition_rows, model, bootstrap_reps, condition_seed
            )
        advantage_ci = {}
        for model in baseline_models:
            differences = np.asarray([
                row["correct"] - row["baselines"][model]["correct"]
                for row in condition_rows
            ])
            advantage_ci[model] = _ci(
                differences, bootstrap_reps,
                _derived_seed(condition_seed, f"{model}:accuracy_advantage"),
            )
        conditions[condition] = {
            "decision_model": decision_summary,
            "constrained_ar": baseline_summaries,
            "h1_accuracy_advantage": {
                model: decision_summary["accuracy"] - summary["accuracy"]
                for model, summary in baseline_summaries.items()
            },
            "h1_accuracy_advantage_ci95": advantage_ci,
            "h2_schema_complexity": {
                model: _split_advantage(condition_rows, "complexity", model,
                                        bootstrap_reps, condition_seed)
                for model in baseline_models
            },
            "h2_candidate_similarity": {
                model: _split_advantage(condition_rows, "similarity", model,
                                        bootstrap_reps, condition_seed)
                for model in baseline_models
            },
        }
    comparisons = {
        "short_vs_long": _pair_comparison(
            rows, "short", "long", bootstrap_reps, bootstrap_seed
        ),
        "semantic_vs_short": _pair_comparison(
            rows, "semantic", "short", bootstrap_reps, bootstrap_seed
        ),
        "semantic_vs_long": _pair_comparison(
            rows, "semantic", "long", bootstrap_reps, bootstrap_seed
        ),
    }
    return {
        "backend": artifact["backend"],
        "model_info": artifact["model_info"],
        "successful_requests": len(rows),
        "bootstrap": {"reps": bootstrap_reps, "seed": bootstrap_seed,
                      "unit": "BFCL task", "interval": "percentile_95"},
        "conditions": conditions,
        "h2_label_sensitivity": comparisons["short_vs_long"],
        "label_condition_comparisons": comparisons,
        "semantic_vs_arbitrary": _semantic_vs_arbitrary(
            rows, bootstrap_reps, bootstrap_seed
        ),
    }


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    result = evaluate(args.artifact, args.bootstrap_reps, args.bootstrap_seed)
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
