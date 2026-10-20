"""Run matched Clef/Qwen3.8-27B structured decisions on When2Call."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from clearml import Task

from qwen_structured_choice import load_qwen_structured_choice
from run_clearml_open_decision_gate import load_clef
from run_jev_bfcl_gate import _checkpoint
from when2call_adapter import build_when2call_cases, request_body, resolve_when2call_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("clef", "qwen38"), required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=120,
                        help="Number of examples; 0 selects all 3,652")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_when2call_data(args.data)
    cases = build_when2call_cases(data_path, args.limit or None, args.seed)
    if args.output is None:
        scope = args.limit or "all"
        args.output = Path(f"when2call_{args.backend}_{scope}.json")
    if args.dry_run:
        print(f"data={data_path} cases={len(cases)}")
        print(json.dumps(request_body(cases[0], args.backend), indent=2)[:6000])
        return

    task = Task.init(
        project_name="Diploma Thesis Multi-Turn UQ",
        task_name=f"When2Call matched choice | {args.backend} | n={args.limit or 'all'}",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    task.add_tags(["when2call", "structured-output", "decision-model", "matched-27b",
                   args.backend])
    task.connect({
        "backend": args.backend,
        "dataset_revision": "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc",
        "output": str(args.output),
        "limit": args.limit,
        "seed": args.seed,
    }, name="experiment")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_properties(0)
    task.get_logger().report_text(
        f"CUDA: {gpu.name}; capability={gpu.major}.{gpu.minor}; "
        f"memory={gpu.total_memory / 2**30:.1f} GiB"
    )
    if args.backend == "clef":
        decide, model_info = load_clef()
    else:
        decide, model_info = load_qwen_structured_choice()

    payload: dict[str, Any] = {
        "method": "When2Call matched bounded response selection",
        "backend": args.backend,
        "model_info": model_info,
        "dataset": {
            "name": "NVIDIA/When2Call",
            "revision": "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc",
            "test_population": 3652,
        },
        "selection": {"seed": args.seed, "limit": args.limit or None, "requests": len(cases)},
        "results": [],
    }
    if args.output.exists():
        previous = json.load(args.output.open())
        if (previous.get("backend") == args.backend
                and previous.get("selection") == payload["selection"]):
            payload = previous
    completed = {row["case_id"] for row in payload["results"] if row.get("error") is None}

    fatal_error: str | None = None
    for index, case in enumerate(cases, 1):
        if case["case_id"] in completed:
            continue
        started = time.perf_counter()
        record: dict[str, Any] = {
            key: case[key] for key in (
                "case_id", "condition", "correct_label", "candidate_labels",
                "candidate_kinds", "candidate_actions", "gold_category", "source",
                "source_id", "tool_count", "official_hallucination_eligible",
            )
        }
        try:
            response = decide(request_body(case, model_info["api_model"]))
            answer = response["answers"]["action"]
            predicted_index = case["candidate_labels"].index(answer["choice"])
            predicted_category = case["candidate_kinds"][predicted_index]
            record.update({
                "choice": answer["choice"],
                "predicted_category": predicted_category,
                "confidence": float(answer["confidence"]),
                "probabilities": {
                    key: float(value) for key, value in answer["probabilities"].items()
                },
                "correct": int(answer["choice"] == case["correct_label"]),
                "tool_hallucination": int(
                    case["official_hallucination_eligible"]
                    and predicted_category == "tool_call"
                ),
                "model_version": response.get("model"),
                "usage": response.get("usage"),
                "model_latency_ms": response.get("latency_ms"),
                "latency_s": time.perf_counter() - started,
                "error": None,
            })
        except Exception as exc:
            record.update({
                "latency_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                fatal_error = record["error"]
        payload["results"].append(record)
        _checkpoint(args.output, payload)
        print(
            f"[{index}/{len(cases)}] {case['case_id']} gold={case['gold_category']} "
            f"pred={record.get('predicted_category', 'ERROR')} "
            f"correct={record.get('correct', '-')} {record['latency_s']:.2f}s",
            flush=True,
        )
        if fatal_error:
            break

    successful = [row for row in payload["results"] if row.get("error") is None]
    accuracy = sum(row["correct"] for row in successful) / len(successful) if successful else 0.0
    eligible = [row for row in successful if row["official_hallucination_eligible"]]
    hallucination_rate = (
        sum(row["tool_hallucination"] for row in eligible) / len(eligible) if eligible else 0.0
    )
    logger = task.get_logger()
    logger.report_scalar("when2call", "successful_requests", len(successful), 0)
    logger.report_scalar("when2call", "accuracy", accuracy, 0)
    logger.report_scalar("when2call", "tool_hallucination_rate", hallucination_rate, 0)
    task.upload_artifact(f"when2call_{args.backend}", artifact_object=str(args.output))
    print(
        f"uploaded {args.output}; successful={len(successful)}/{len(cases)} "
        f"accuracy={accuracy:.3f} tool_hallucination={hallucination_rate:.3f}",
        flush=True,
    )
    if fatal_error:
        raise RuntimeError(f"Run stopped after fatal GPU error: {fatal_error}")


if __name__ == "__main__":
    main()
