"""Evaluate SAUP, UProp and HTC on a JSON file of agent trajectories.

Expected input:
{
  "rows": [{
    "wrong_valid": 0,
    "trajectory_steps": [{
      "token_confidences": [...],
      "top1_confidences": [...],
      "topk_confidences": [...],
      "uncertainty": 0.4,
      "situation_weight": 1.2
    }, ...],
    "uprop_tdps": [{
      "intrinsic_uncertainties": [...],
      "predecessor_distances": [[], [[...]], ...]
    }, ...]
  }]
}

HTC is a supervised calibrator, so its score is always out-of-fold.  SAUP and
UProp are raw paper scores.  Rows missing a method's required fields are
reported as unavailable rather than silently approximated.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import risk_at_cov
from trajectory_baselines import HTC_FEATURE_NAMES, htc_features, saup_score, uprop_score


def _metrics(name, y, score):
    print(f"{name:20s} {roc_auc_score(y, score):.3f}  "
          f"{average_precision_score(y, score):.3f}  "
          f"{risk_at_cov(score, y):.3f}")


def _htc_oof(rows, y, penalty):
    x = np.asarray([
        [htc_features(r["trajectory_steps"])[f] for f in HTC_FEATURE_NAMES]
        for r in rows
    ])
    if penalty == "l1":
        model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=4000)
    else:
        model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=4000)
    clf = make_pipeline(StandardScaler(), model)
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("HTC cross-validation needs at least two rows per class")
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=2026)
    return cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]


def main(path):
    blob = json.load(open(path))
    # Multi-turn runner keeps action rows for error localization and separate
    # one-row-per-trajectory records for trajectory-level methods.
    rows = blob.get("trajectory_records", blob["rows"])
    y = np.asarray([r["wrong_valid"] for r in rows], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("evaluation requires both correct and wrong-valid rows")
    print(f"n={len(rows)} wrong_valid={y.mean():.3f}")
    print(f"{'method':20s} AUROC AUPRC Risk@80%")

    if all("trajectory_steps" in r for r in rows):
        for penalty, label in [("l2", "HTC-Full (OOF)"),
                               ("l1", "HTC-Reduced (OOF)")]:
            _metrics(label, y, _htc_oof(rows, y, penalty))
    else:
        print("HTC                  unavailable: missing trajectory_steps")

    if all("trajectory_steps" in r and all(
            "uncertainty" in s and "situation_weight" in s
            for s in r["trajectory_steps"]) for r in rows):
        scores = np.asarray([saup_score(
            [s["uncertainty"] for s in r["trajectory_steps"]],
            [s["situation_weight"] for s in r["trajectory_steps"]],
        ) for r in rows])
        _metrics("SAUP", y, scores)
    else:
        print("SAUP                 unavailable: missing step uncertainty/weight")

    if all("uprop_tdps" in r for r in rows):
        scores = np.asarray([uprop_score(r["uprop_tdps"]) for r in rows])
        _metrics("UProp", y, scores)
    else:
        print("UProp                unavailable: missing uprop_tdps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    main(parser.parse_args().path)
