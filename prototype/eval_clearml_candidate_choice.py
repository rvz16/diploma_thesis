"""Evaluate a candidate-choice artifact inside ClearML storage context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearml import Task

from eval_candidate_choice import evaluate


def _report_scalars(task: Task, result: dict) -> None:
    logger = task.get_logger()
    for condition, block in result["conditions"].items():
        decision = block["decision_model"]
        for metric in ("accuracy", "brier", "ece_10", "adaptive_ece_10",
                       "log_loss", "multiclass_brier", "mean_p_gold", "mean_latency_s"):
            logger.report_scalar(f"decision_{condition}", metric, decision[metric], 0)
        for model, metrics in block["constrained_ar"].items():
            series = model.replace("Qwen/Qwen2.5-", "qwen_").replace("-Instruct", "")
            for metric in ("accuracy", "brier", "ece_10", "adaptive_ece_10", "log_loss"):
                logger.report_scalar(f"ar_{condition}_{series}", metric, metrics[metric], 0)
    for metric, value in result["h2_label_sensitivity"].items():
        if metric != "n":
            logger.report_scalar("label_sensitivity", metric, value, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("candidate_choice_metrics.json"))
    args = parser.parse_args()

    task = Task.init(
        project_name="Diploma Thesis Multi-Turn UQ",
        task_name=f"Evaluate candidate choice | {args.source_task[:8]}",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    source = Task.get_task(task_id=args.source_task)
    artifact_path = source.artifacts[args.artifact_name].get_local_copy()
    result = evaluate(Path(artifact_path))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    _report_scalars(task, result)
    task.upload_artifact("candidate_choice_metrics", artifact_object=str(args.output))


if __name__ == "__main__":
    main()

