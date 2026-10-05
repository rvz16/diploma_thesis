"""ClearML entry point for the first GPU BFCL multi-turn constrained rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Task

from bfcl_multiturn import DEFAULT_DATA, run_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B-Instruct")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--output", default="bfcl_multiturn_base_qwen3_14b_smoke.json")
    args = p.parse_args()
    task = Task.init(
        project_name="Diploma Thesis / Structured Output UQ",
        task_name="BFCL-multi-turn-base | Qwen3-14B | constrained CP | smoke-20",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    task.add_tags(["bfcl-multi-turn", "stateful", "constrained-decoding", "cp", "smoke"])
    task.connect(vars(args), name="experiment")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this ClearML experiment.")
    task.get_logger().report_text(
        f"CUDA: {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
    )
    payload = run_evaluation(args.model, args.limit, DEFAULT_DATA, None, True, args.output)
    records = payload["trajectory_records"]
    if records:
        failure_rate = sum(r["trajectory_failure"] for r in records) / len(records)
        task.get_logger().report_scalar("BFCL multi-turn", "failure_rate", failure_rate, 0)
        task.get_logger().report_scalar("BFCL multi-turn", "trajectories", len(records), 0)
    task.upload_artifact("bfcl_multiturn_rollout", artifact_object=args.output)
    print(f"uploaded ClearML artifact: {args.output}")


if __name__ == "__main__":
    main()
