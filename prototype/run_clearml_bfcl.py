"""Reproducible ClearML entry point for the BFCL constrained-decoding smoke test.

The default is deliberately small: it verifies CUDA, model loading, the pinned
BFCL V4 slice and artifact upload before any longer GPU run is queued.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from clearml import Task

from run_bfcl import main as run_bfcl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="multiple")
    p.add_argument("--model", default="Qwen/Qwen3-14B-Instruct")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--data-version", choices=("v3", "v4"), default="v4")
    p.add_argument("--output", default="clearml_bfcl_v4_multiple_qwen3_14b_smoke.json")
    return p.parse_args()


def main():
    args = parse_args()
    task = Task.init(
        project_name="Diploma Thesis / Structured Output UQ",
        task_name="BFCL-V4-multiple | Qwen3-14B | constrained CP | smoke-20",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    task.add_tags(["bfcl-v4", "multiple", "constrained-decoding", "cp", "smoke"])
    task.connect(vars(args), name="experiment")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This task requires a CUDA GPU; no CUDA device was detected.")
    task.get_logger().report_text(
        f"CUDA device: {torch.cuda.get_device_name(0)}; "
        f"memory={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
    )

    # The dataset travels with the task as a pinned project-local input.
    os.environ.setdefault("BFCL_DATA_DIR", str(Path(__file__).parent / "data"))
    result = run_bfcl(
        args.category, args.model, args.output, limit=args.limit,
        cache="id2str_qwen3_14b_instruct.pkl", dtype=torch.bfloat16,
        data_version=args.data_version,
    )
    rows = result["rows"]
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    task.get_logger().report_scalar("BFCL", "accuracy", acc, iteration=0)
    task.get_logger().report_scalar("BFCL", "wrong_valid_rate", 1 - acc, iteration=0)
    task.get_logger().report_scalar(
        "BFCL", "valid_json_rate", sum(r["valid_json"] for r in rows) / n, iteration=0)
    task.upload_artifact("bfcl_features", artifact_object=args.output)
    print(f"ClearML artifact uploaded: {args.output}")


if __name__ == "__main__":
    main()
