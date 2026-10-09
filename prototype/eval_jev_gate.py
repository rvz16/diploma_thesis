"""Evaluate Jev's native execute/review distribution on BFCL actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from critical_errors import critical_error_metrics


def _key(row):
    return f"{row['trajectory_id']}:{row['turn']}:{row['step']}"


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if mask.any():
            value += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return value


def _rank_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def main(source: Path, jev_path: Path) -> None:
    blob = json.load(source.open())
    jev = json.load(jev_path.open())
    source_rows = {_key(r): r for r in blob["rows"] if r.get("python_call") is not None}
    successful = [r for r in jev["results"] if r.get("error") is None]
    if not successful:
        raise RuntimeError("The Jev artifact has no successful API responses")
    rows = [source_rows[r["action_key"]] for r in successful]
    y = np.asarray([r["wrong_valid"] for r in rows], dtype=int)
    scores = {
        "Jev P(review)": np.asarray([
            float(r["probabilities"].get("review", 1.0 - r["probabilities"]["execute"]))
            for r in successful
        ]),
        "G-NLL-SMT": np.asarray([r["gnll_smt"] for r in rows], dtype=float),
        "CP max": np.asarray([r["cp_max"] for r in rows], dtype=float),
        "CP arg value": np.asarray([r["cp_argvalue"] for r in rows], dtype=float),
    }
    print(f"n={len(y)} wrong_valid={int(y.sum())} prevalence={y.mean():.3f} "
          f"api_failures={len(jev['results']) - len(successful)}")
    print("method          AUROC  AUPRC  Brier  ECE(5)  risk@20  risk@50  risk@80")
    for name, score in scores.items():
        m = _rank_metrics(y, score)
        risks = [critical_error_metrics(y, score, c)["selective_risk"]
                 for c in (0.2, 0.5, 0.8)]
        if name == "Jev P(review)":
            brier = f"{brier_score_loss(y, score):.3f}"
            ece = f"{_ece(y, score):.3f}"
        else:
            # G-NLL and CP are ranking scores, not calibrated probabilities.
            brier = ece = "  —  "
        print(f"{name:15s} {m['auroc']:.3f}  {m['auprc']:.3f}  {brier:>5s}  "
              f"{ece:>6s}   {risks[0]:.3f}    {risks[1]:.3f}    {risks[2]:.3f}")

    p_review = scores["Jev P(review)"]
    accepted = p_review < 0.5
    print(f"\nJev native 0.5 gate: coverage={accepted.mean():.3f} "
          f"accepted_wrong={int(y[accepted].sum())}/{int(accepted.sum())} "
          f"selective_risk={y[accepted].mean() if accepted.any() else float('nan'):.3f}")
    latency = np.asarray([r["latency_s"] for r in successful], dtype=float)
    print(f"latency seconds mean={latency.mean():.3f} median={np.median(latency):.3f} "
          f"p95={np.quantile(latency, 0.95):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("jev", type=Path)
    args = p.parse_args()
    main(args.source, args.jev)
