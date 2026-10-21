"""Instrument complete API-Bank calls with CP and the original logit baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from apibank_adapter import DEFAULT_DATA, build_tasks, exact_correct, schema_valid
from eval_apibank_cp import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("apibank_level1_qwen25_3b_cp.json"))
    parser.add_argument("--max-string-chars", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-clearml", action="store_true")
    return parser.parse_args()


def checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    args = parse_args()
    tasks, manifest = build_tasks(args.data, args.limit or None, args.seed)
    if not tasks:
        raise RuntimeError("No compatible API-Bank calls")
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        print(f"first_task={tasks[0]['id']} tools={len(tasks[0]['tools'])}")
        return

    import torch
    from huggingface_hub import snapshot_download
    from fc_constrained import ConstrainedFC
    from run_poc import features_from_steps

    task = None
    if not args.no_clearml:
        from clearml import Task
        task = Task.init(
            project_name="Diploma Thesis Multi-Turn UQ",
            task_name="API-Bank level-1 | Qwen2.5-3B | CP + logit baselines",
            task_type=Task.TaskTypes.testing,
            reuse_last_task_id=False,
        )
        task.add_tags(["api-bank", "cp", "logit-baselines", "full-call", "teacher-forced"])
        task.connect({
            "model": args.model, "revision": args.revision, "limit": args.limit,
            "seed": args.seed, "max_string_chars": args.max_string_chars,
        }, name="experiment")
        if not torch.cuda.is_available():
            raise RuntimeError("ClearML run requires a CUDA GPU")
        task.get_logger().report_text(
            f"GPU {torch.cuda.get_device_name(0)}; "
            f"memory={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
        )

    model_path = snapshot_download(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    cache_tag = hashlib.sha256(f"{args.model}:{args.revision}".encode()).hexdigest()[:16]
    fc = ConstrainedFC(
        model_path, id2str_cache=f"id2str_apibank_{cache_tag}.pkl",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    fc.MAX_STR = args.max_string_chars
    selection = {
        "manifest": manifest, "model": args.model, "revision": args.revision,
        "max_string_chars": args.max_string_chars,
    }
    payload = {"model": args.model, "benchmark": "API-Bank level-1 scalar subset",
               "selection": selection, "rows": []}
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old.get("selection") != selection:
            raise ValueError("Output file belongs to a different experiment")
        payload = old
    completed = {row["id"] for row in payload["rows"] if row.get("error") is None}
    print("MANIFEST " + json.dumps(manifest), flush=True)
    fatal_error = None
    for index, entry in enumerate(tasks, 1):
        if entry["id"] in completed:
            continue
        started = time.perf_counter()
        row = {key: entry[key] for key in ("id", "dialogue_id", "position", "family", "gold")}
        try:
            decoded = fc.generate_schema(entry["system"], entry["user"], entry["tools"])
            valid = schema_valid(decoded.parsed, entry["tools"])
            correct = exact_correct(decoded.parsed, entry)
            row.update({
                "pred": decoded.text, "parsed": decoded.parsed,
                "correct": int(correct), "schema_valid": int(valid),
                "valid_json": int(decoded.parsed is not None),
                "wrong_valid": int(valid and not correct),
                "schema_complexity": len(entry["tools"]) + sum(len(p) for _, p in entry["tools"]),
                "token_steps": [asdict(step) for step in decoded.steps],
                **features_from_steps(decoded.steps),
                "error": None,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (torch.cuda.OutOfMemoryError, RuntimeError)):
                fatal_error = row["error"]
        row["latency_s"] = time.perf_counter() - started
        payload["rows"] = [old for old in payload["rows"] if old["id"] != row["id"]] + [row]
        checkpoint(args.output, payload)
        print(
            f"[{index}/{len(tasks)}] {entry['id']} valid={row.get('schema_valid', '-')} "
            f"correct={row.get('correct', '-')} error={row['error']} "
            f"latency={row['latency_s']:.2f}s", flush=True,
        )
        if fatal_error:
            break
    metrics = evaluate(payload["rows"])
    payload["metrics"] = metrics
    checkpoint(args.output, payload)
    print("METRICS " + json.dumps(metrics), flush=True)
    if task:
        task.upload_artifact("apibank_cp_features", artifact_object=str(args.output))
        task.upload_artifact("apibank_cp_metrics", artifact_object=metrics)
        for key in ("strict_call_accuracy", "wrong_valid", "schema_valid", "invalid"):
            if metrics.get(key) is not None:
                task.get_logger().report_scalar("API-Bank", key, metrics[key], 0)
        for name, values in metrics.get("oof_detectors", {}).items():
            if values["auroc"] is not None:
                task.get_logger().report_scalar("OOF AUROC", name, values["auroc"], 0)
    if fatal_error:
        raise RuntimeError(fatal_error)
    if metrics["inference_errors"]:
        raise RuntimeError(f"{metrics['inference_errors']} inference errors; inspect artifact")


if __name__ == "__main__":
    main()
