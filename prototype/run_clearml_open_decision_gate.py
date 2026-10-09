"""Evaluate open Jev-compatible decision models on recorded BFCL actions.

The two supported backends are Cloudflare Clef and PostHog Jeeves.  Both run
from public weights on the ClearML GPU; no hosted-model API key is involved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
from clearml import Task
from huggingface_hub import snapshot_download

from run_jev_bfcl_gate import (
    DEFAULT_DATA,
    DEFAULT_INPUT,
    GATE_CRITERIA,
    GATE_INSTRUCTIONS,
    _checkpoint,
    action_key,
    build_state,
    load_entries,
    select_actions,
)


CLEF_MODEL = "Cloudflare/clef"
CLEF_REVISION = "ed3eed331870db2eff4b0db01237128ede8a00ce"
JEEVES_MODEL = "PostHog/jeeves"
JEEVES_REVISION = "c754d6794ae04012b46eac3c6a8623521add7387"
JEEVES_REPO = "https://github.com/PostHog/jeeves.git"
JEEVES_CODE_REVISION = "3f948dec68187ed3ced9152ed3d84b73e498665c"


def request_body(state: dict[str, Any], model: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": {
            "gate": {
                "type": "choice",
                "instructions": GATE_INSTRUCTIONS,
                "criteria": GATE_CRITERIA,
            }
        },
    }
    if options:
        body["options"] = options
    return body


def load_clef() -> tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]]:
    model_path = snapshot_download(CLEF_MODEL, revision=CLEF_REVISION)
    sys.path.insert(0, model_path)
    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(model_path, device="cuda")

    def decide(body: dict[str, Any]) -> dict[str, Any]:
        return systemone(model, processor, body)

    return decide, {
        "model": CLEF_MODEL,
        "api_model": "clef",
        "weights_revision": CLEF_REVISION,
        "precision": "bfloat16",
    }


def _clone_jeeves() -> Path:
    repo = Path(tempfile.gettempdir()) / f"jeeves-{JEEVES_CODE_REVISION[:12]}"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", JEEVES_REPO, str(repo)],
            check=True,
        )
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", JEEVES_CODE_REVISION], check=True)
    return repo


def load_jeeves(max_think: int) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]]:
    repo = _clone_jeeves()
    model_path = Path(snapshot_download(JEEVES_MODEL, revision=JEEVES_REVISION))
    sys.path.insert(0, str(repo))
    from inference.engine import Engine
    from inference.serve import Server
    from inference.types import Options

    # A100 has compute capability 8.0, below Jeeves' >=8.9 FP8 requirement.
    # BF16 is still comfortably within an 80-GiB worker.  One gate question is
    # evaluated per request, so a single cache row is sufficient.
    engine = Engine(
        str(model_path),
        str(model_path / "drafter_k4.safetensors"),
        block=4,
        precision="bf16",
        max_rows=1,
        max_len=16384,
        device="cuda",
    )
    defaults = Options(max_think=max_think, return_reasoning=False)
    server = Server(engine, defaults, {
        "model": JEEVES_MODEL,
        "revision": JEEVES_REVISION,
        "precision": "bf16",
    })

    def decide(body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        body["options"] = {
            "think": True,
            "max_think": max_think,
            "return_reasoning": False,
        }
        return server.systemone(body)

    return decide, {
        "model": JEEVES_MODEL,
        "api_model": "jeeves-latest",
        "weights_revision": JEEVES_REVISION,
        "code_revision": JEEVES_CODE_REVISION,
        "precision": "bfloat16",
        "max_think": max_think,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("clef", "jeeves"), required=True)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-think", type=int, default=768,
                   help="Jeeves reasoning cap; ignored by Clef")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = Path(f"bfcl_multiturn_qwen3_14b_{args.backend}_gate_smoke.json")

    task = Task.init(
        project_name="Diploma Thesis Multi-Turn UQ",
        task_name=f"BFCL action gate | {args.backend} | open weights | smoke-{args.limit}",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    task.add_tags(["bfcl-multi-turn", "decision-model", "open-weights", args.backend])
    task.connect({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 name="experiment")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_properties(0)
    task.get_logger().report_text(
        f"CUDA: {gpu.name}; capability={gpu.major}.{gpu.minor}; memory={gpu.total_memory / 2**30:.1f} GiB"
    )

    blob = json.load(args.input.open())
    entries = load_entries(args.data_dir)
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for row in blob["rows"]:
        trajectories.setdefault(row["trajectory_id"], []).append(row)
    for rows in trajectories.values():
        rows.sort(key=lambda r: (r["turn"], r["step"]))
    selected = select_actions(blob["rows"], args.limit or None, args.seed)

    decide, model_info = load_clef() if args.backend == "clef" else load_jeeves(args.max_think)
    payload: dict[str, Any] = {
        "method": f"{args.backend} Choice pre-execution gate",
        "backend": args.backend,
        "model_info": model_info,
        "source_artifact": str(args.input),
        "selection": {
            "seed": args.seed,
            "limit": args.limit or None,
            "population_actions": sum(r.get("python_call") is not None for r in blob["rows"]),
        },
        "gate": {"instructions": GATE_INSTRUCTIONS, "criteria": GATE_CRITERIA},
        "results": [],
    }
    fatal_error: str | None = None
    for index, row in enumerate(selected, 1):
        state = build_state(
            entries[row["trajectory_id"]], trajectories[row["trajectory_id"]], row, args.data_dir
        )
        started = time.perf_counter()
        record: dict[str, Any] = {
            "action_key": action_key(row),
            "trajectory_id": row["trajectory_id"],
            "turn": row["turn"],
            "step": row["step"],
            "state_chars": len(json.dumps(state)),
            "prior_action_count": len(state["prior_executed_actions"]),
            "tool_count": len(state["available_tools"]),
        }
        try:
            response = decide(request_body(
                state,
                model_info["api_model"],
                {"think": True, "max_think": args.max_think,
                 "return_reasoning": False} if args.backend == "jeeves" else None,
            ))
            answer = response["answers"]["gate"]
            record.update({
                "choice": answer["choice"],
                "confidence": float(answer["confidence"]),
                "probabilities": {k: float(v) for k, v in answer["probabilities"].items()},
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
            f"[{index}/{len(selected)}] {record['action_key']} "
            f"{record.get('choice', 'ERROR')} {record['latency_s']:.2f}s",
            flush=True,
        )
        if fatal_error:
            break

    ok = sum(r.get("error") is None for r in payload["results"])
    task.get_logger().report_scalar("gate", "successful_actions", ok, 0)
    task.get_logger().report_scalar("gate", "attempted_actions", len(payload["results"]), 0)
    task.upload_artifact(f"bfcl_{args.backend}_gate", artifact_object=str(args.output))
    print(f"uploaded {args.output}; successful={ok}/{len(payload['results'])}")
    if fatal_error:
        raise RuntimeError(f"Gate run stopped after fatal model error: {fatal_error}")


if __name__ == "__main__":
    main()
