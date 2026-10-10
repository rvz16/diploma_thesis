"""Fetch paired Clef/Qwen artifacts from ClearML and evaluate them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearml import Task

from eval_matched_structured_choice import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clef-task", required=True)
    parser.add_argument("--qwen-task", required=True)
    parser.add_argument("--clef-artifact", default="bfcl_candidate_choice_clef")
    parser.add_argument("--qwen-artifact", default="bfcl_candidate_choice_qwen38")
    parser.add_argument("--condition", default="short")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("matched_structured_choice_metrics.json"))
    args = parser.parse_args()

    task = Task.init(
        project_name="Diploma Thesis Multi-Turn UQ",
        task_name="Evaluate matched Clef vs Qwen3.8 structured choice",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    clef = Task.get_task(task_id=args.clef_task).artifacts[args.clef_artifact].get_local_copy()
    qwen = Task.get_task(task_id=args.qwen_task).artifacts[args.qwen_artifact].get_local_copy()
    result = evaluate(
        Path(clef), Path(qwen), args.condition, args.bootstrap_reps, args.bootstrap_seed
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    task.upload_artifact("matched_structured_choice_metrics", artifact_object=str(args.output))
    logger = task.get_logger()
    logger.report_scalar("accuracy", "clef", result["clef"]["accuracy"], 0)
    logger.report_scalar("accuracy", "qwen_structured", result["qwen_structured"]["accuracy"], 0)
    logger.report_scalar(
        "paired", "accuracy_difference_clef_minus_qwen",
        result["paired"]["accuracy_difference_clef_minus_qwen"], 0,
    )
    logger.report_scalar(
        "paired", "same_candidate_rate", result["paired"]["same_candidate_rate"], 0
    )


if __name__ == "__main__":
    main()

