"""Run open decision models on paired BFCL bounded candidate-selection cases."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from clearml import Task

from bfcl_candidate_choice import (
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_FEATURES_3B,
    LABEL_CONDITIONS,
    build_encoded_cases,
    request_body,
)
from run_clearml_open_decision_gate import load_clef, load_jeeves, load_laya
from run_jev_bfcl_gate import _checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("clef", "jeeves", "laya"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--features-3b", type=Path, default=DEFAULT_FEATURES_3B)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=40,
                        help="Number of unique BFCL tasks; 0 selects the full population")
    parser.add_argument("--conditions", default="short,long")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-think", type=int, default=768)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    if not conditions or any(item not in LABEL_CONDITIONS for item in conditions):
        raise ValueError(f"--conditions must contain values from {LABEL_CONDITIONS}")
    if args.output is None:
        scope = args.limit if args.limit else "all"
        args.output = Path(f"bfcl_candidate_choice_{args.backend}_{scope}.json")

    cases = build_encoded_cases(
        args.data_dir,
        args.features,
        args.features_3b,
        conditions=conditions,
        seed=args.seed,
        limit=args.limit or None,
    )
    if args.dry_run:
        print(f"unique_tasks={len({case['case_id'] for case in cases})} requests={len(cases)}")
        print(json.dumps(request_body(cases[0], args.backend), indent=2)[:6000])
        return

    task = Task.init(
        project_name="Diploma Thesis Multi-Turn UQ",
        task_name=(f"BFCL candidate choice | {args.backend} | "
                   f"n={args.limit or 'all'}x{len(conditions)}"),
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    task.add_tags(["bfcl", "candidate-choice", "h1-h2-h3", args.backend, *conditions])
    task.connect({
        "backend": args.backend,
        "data_dir": str(args.data_dir),
        "features": str(args.features),
        "features_3b": str(args.features_3b),
        "output": str(args.output),
        "limit": args.limit,
        "conditions": ",".join(conditions),
        "seed": args.seed,
        "max_think": args.max_think,
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
    elif args.backend == "jeeves":
        decide, model_info = load_jeeves(args.max_think)
    else:
        decide, model_info = load_laya()
    payload: dict[str, Any] = {
        "method": "BFCL bounded candidate selection",
        "backend": args.backend,
        "model_info": model_info,
        "source_features": str(args.features),
        "selection": {
            "seed": args.seed,
            "limit": args.limit or None,
            "conditions": list(conditions),
            "unique_tasks": len({case["case_id"] for case in cases}),
            "requests": len(cases),
        },
        "results": [],
    }
    if args.output.exists():
        previous = json.load(args.output.open())
        if (previous.get("backend") == args.backend
                and previous.get("selection") == payload["selection"]):
            payload = previous
    completed = {
        (row["case_id"], row["condition"])
        for row in payload["results"] if row.get("error") is None
    }

    fatal_error: str | None = None
    for index, case in enumerate(cases, 1):
        key = (case["case_id"], case["condition"])
        if key in completed:
            continue
        started = time.perf_counter()
        record: dict[str, Any] = {
            "case_id": case["case_id"],
            "condition": case["condition"],
            "correct_label": case["correct_label"],
            "candidate_labels": case["candidate_labels"],
            "candidate_kinds": [candidate["kind"] for candidate in case["candidates"]],
            "candidate_actions": [candidate["action"] for candidate in case["candidates"]],
            "schema_complexity": case["schema_complexity"],
            "near_tool_similarity": case["near_tool_similarity"],
            "baselines": case["baselines"],
        }
        try:
            response = decide(request_body(case, model_info["api_model"]))
            answer = response["answers"]["action"]
            record.update({
                "choice": answer["choice"],
                "confidence": float(answer["confidence"]),
                "probabilities": {key: float(value)
                                  for key, value in answer["probabilities"].items()},
                "correct": int(answer["choice"] == case["correct_label"]),
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
            if isinstance(exc, (torch.cuda.OutOfMemoryError, RuntimeError)):
                fatal_error = record["error"]
        payload["results"].append(record)
        _checkpoint(args.output, payload)
        print(
            f"[{index}/{len(cases)}] {case['case_id']}:{case['condition']} "
            f"{record.get('choice', 'ERROR')} correct={record.get('correct', '-')} "
            f"{record['latency_s']:.2f}s",
            flush=True,
        )
        if fatal_error:
            break

    successful = [row for row in payload["results"] if row.get("error") is None]
    accuracy = sum(row["correct"] for row in successful) / len(successful) if successful else 0.0
    logger = task.get_logger()
    logger.report_scalar("candidate_choice", "successful_requests", len(successful), 0)
    logger.report_scalar("candidate_choice", "attempted_requests", len(payload["results"]), 0)
    logger.report_scalar("candidate_choice", "accuracy", accuracy, 0)
    task.upload_artifact(f"bfcl_candidate_choice_{args.backend}", artifact_object=str(args.output))
    print(f"uploaded {args.output}; successful={len(successful)}/{len(cases)} accuracy={accuracy:.3f}")
    if fatal_error:
        raise RuntimeError(f"Candidate run stopped after fatal model error: {fatal_error}")


if __name__ == "__main__":
    main()
