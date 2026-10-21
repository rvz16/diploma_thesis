"""CP/logit detection metrics with dialogue-grouped out-of-fold calibration."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOGIT_FEATURES = ["gnll", "gnll_smt", "max_tok_unc", "avg_tok_unc"]
CP_FEATURES = ["pc", "pc_sem", "cp_max", "cp_function", "cp_argvalue"]


def score_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    order = np.argsort(scores, kind="stable")
    return {
        "auroc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        "auprc": float(average_precision_score(y, scores)) if y.sum() else None,
        "selective_risk": {
            str(coverage): float(y[order[:max(1, int(np.ceil(len(y) * coverage)))]].mean())
            for coverage in (0.2, 0.5, 0.8)
        },
    }


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row.get("error") is None]
    valid = [row for row in attempted if row["schema_valid"]]
    result: dict[str, Any] = {
        "attempted": len(rows),
        "successful_inference": len(attempted),
        "inference_errors": len(rows) - len(attempted),
        "schema_valid": len(valid),
        "invalid": len(attempted) - len(valid),
        "strict_call_accuracy": (
            sum(row["correct"] for row in attempted) / len(attempted) if attempted else None
        ),
        "wrong_valid": sum(row["wrong_valid"] for row in valid),
        "detection_population": "schema-valid calls only; invalid calls reported separately",
        "correctness_metric": "strict reference name/arguments, not official execution accuracy",
    }
    if not valid:
        return result
    y = np.asarray([row["wrong_valid"] for row in valid], dtype=int)
    groups = np.asarray([row["dialogue_id"] for row in valid])
    result["raw_scores"] = {
        feature: score_metrics(y, np.asarray([row[feature] for row in valid]))
        for feature in LOGIT_FEATURES + CP_FEATURES + ["avg_sem_tok_unc"]
    }
    result["per_api"] = {
        family: {
            "n": sum(row["family"] == family for row in attempted),
            "correct": sum(row["correct"] for row in attempted if row["family"] == family),
        }
        for family in sorted({row["family"] for row in attempted})
    }
    minority = min(Counter(y).values()) if len(set(y)) == 2 else 0
    splits = min(5, minority, len(set(groups)))
    if splits < 2:
        result["oof_unavailable"] = "Need at least two examples per class and two dialogues"
        return result
    folds = list(StratifiedGroupKFold(
        n_splits=splits, shuffle=True, random_state=2026
    ).split(np.zeros(len(y)), y, groups))
    if any(len(set(y[train])) < 2 for train, _ in folds):
        result["oof_unavailable"] = "A grouped training fold has only one class"
        return result
    result["cv"] = {
        "folds": splits, "split": "StratifiedGroupKFold", "group": "source dialogue",
        "seed": 2026,
        "note": "Target-domain OOF detector fit; not frozen cross-benchmark transfer",
    }
    result["oof_detectors"] = {}
    for name, features in {
        "CP-only": CP_FEATURES,
        "logit-baselines": LOGIT_FEATURES,
        "logit-plus-CP": LOGIT_FEATURES + CP_FEATURES,
    }.items():
        matrix = np.asarray([[row[feature] for feature in features] for row in valid])
        probabilities = np.zeros(len(y))
        for train, test in folds:
            assert not set(groups[train]) & set(groups[test])
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            model.fit(matrix[train], y[train])
            probabilities[test] = model.predict_proba(matrix[test])[:, 1]
        result["oof_detectors"][name] = {
            **score_metrics(y, probabilities),
            "brier": float(brier_score_loss(y, probabilities)),
        }
    return result
