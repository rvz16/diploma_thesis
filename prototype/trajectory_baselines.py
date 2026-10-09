"""Trajectory-level uncertainty baselines: SAUP, UProp, and HTC.

The implementations follow the equations in the papers shipped with this
repository.  They deliberately operate on *agent steps*, not arbitrary chunks
of a single JSON function call.  A one-step trajectory is accepted, but SAUP
then reduces to its single-step uncertainty, UProp has no extrinsic term, and
most HTC dynamics features are zero.

All uncertainty scores use the convention "larger means more uncertain".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np


EPS = 1e-12


# ---------------------------------------------------------------------------
# Shared single-step uncertainty
# ---------------------------------------------------------------------------
def normalized_nll(token_probabilities: Sequence[float]) -> float:
    """Length-normalized response NLL (SAUP Eq. 3 / UProp PE estimate)."""
    p = np.asarray(token_probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("token_probabilities must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("token probabilities must be finite and in (0, 1]")
    return float(-np.log(np.maximum(p, EPS)).mean())


def mc_response_uncertainty(
    response_token_probabilities: Sequence[Sequence[float]],
) -> float:
    """Monte-Carlo estimate E[-log p(response) / response length]."""
    if not response_token_probabilities:
        raise ValueError("at least one response sample is required")
    return float(np.mean([
        normalized_nll(sample) for sample in response_token_probabilities
    ]))


# ---------------------------------------------------------------------------
# SAUP (Situation-Awareness Uncertainty Propagation)
# ---------------------------------------------------------------------------
def saup_step_uncertainty(
    thought_samples: Sequence[Sequence[float]] | None = None,
    action_samples: Sequence[Sequence[float]] | None = None,
) -> float:
    """SAUP step uncertainty U_n = U_n^T + U_n^A.

    Each argument contains MC response samples, and each response sample is a
    sequence of emitted-token probabilities.  Either thought or action may be
    omitted for agent frameworks that expose only one of them.
    """
    if thought_samples is None and action_samples is None:
        raise ValueError("thought_samples or action_samples is required")
    score = 0.0
    if thought_samples is not None:
        score += mc_response_uncertainty(thought_samples)
    if action_samples is not None:
        score += mc_response_uncertainty(action_samples)
    return score


def simple_uncertainty_propagation(
    step_uncertainties: Sequence[float], method: str = "rms",
) -> float:
    """Unweighted arithmetic/geometric/RMS propagation baselines."""
    u = _finite_vector(step_uncertainties, "step_uncertainties")
    if np.any(u < 0.0):
        raise ValueError("uncertainties must be non-negative")
    if method == "arithmetic":
        return float(u.mean())
    if method == "geometric":
        return float(np.exp(np.log(np.maximum(u, EPS)).mean()))
    if method == "rms":
        return float(np.sqrt(np.mean(u ** 2)))
    raise ValueError("method must be 'arithmetic', 'geometric', or 'rms'")


def saup_score(
    step_uncertainties: Sequence[float],
    situation_weights: Sequence[float],
) -> float:
    """Weighted RMS propagation from SAUP Eq. 1.

    ``situation_weights`` can be the paper's plain-distance surrogate
    (inquiry drift + inference gap) or learned CHMM state weights.  Learning a
    CHMM is intentionally outside this pure aggregation function.
    """
    u = _finite_vector(step_uncertainties, "step_uncertainties")
    w = _finite_vector(situation_weights, "situation_weights")
    if u.shape != w.shape:
        raise ValueError("uncertainties and weights must have equal length")
    if np.any(u < 0.0) or np.any(w < 0.0):
        raise ValueError("uncertainties and weights must be non-negative")
    return float(np.sqrt(np.mean((w * u) ** 2)))


def saup_distance_weights(
    inquiry_drift: Sequence[float], inference_gap: Sequence[float],
) -> np.ndarray:
    """SAUP-D plain-distance surrogate W_n = Da_n + Do_n."""
    da = _finite_vector(inquiry_drift, "inquiry_drift")
    do = _finite_vector(inference_gap, "inference_gap")
    if da.shape != do.shape:
        raise ValueError("drift and gap must have equal length")
    if np.any(da < 0.0) or np.any(do < 0.0):
        raise ValueError("distances must be non-negative")
    return da + do


# ---------------------------------------------------------------------------
# UProp
# ---------------------------------------------------------------------------
def uprop_kernel(distance: float, sharpness: int) -> float:
    """Gaussian neighborhood kernel K_tau(d), UProp Eq. 7.

    The paper defines K_tau(x) = (phi(x))**tau and uses tau=N, where N is the
    number of per-step samples.  This function follows that published formula.
    """
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and non-negative")
    if sharpness < 1:
        raise ValueError("sharpness must be positive")
    gaussian_density = math.exp(-0.5 * distance * distance) / math.sqrt(
        2.0 * math.pi
    )
    return gaussian_density ** sharpness


def uprop_pmi(
    predecessor_distances: Sequence[float],
    kernel: Callable[[float, int], float] = uprop_kernel,
) -> float:
    """Trajectory-dependent PMI approximation from UProp Eq. 8.

    Distances compare the realized predecessor decision with the N alternative
    decisions sampled at that predecessor step.
    """
    d = _finite_vector(predecessor_distances, "predecessor_distances")
    if np.any(d < 0.0):
        raise ValueError("distances must be non-negative")
    n = len(d)
    neighborhood_mass = sum(kernel(float(x), n) for x in d)
    return -math.log(max(neighborhood_mass, EPS))


def uprop_tdp_score(
    intrinsic_uncertainties: Sequence[float],
    predecessor_distances: Sequence[Sequence[Sequence[float]]],
    *,
    length_normalize: bool = True,
    kernel: Callable[[float, int], float] = uprop_kernel,
) -> float:
    """Uncertainty for one trajectory-dependent decision process (TDP).

    ``predecessor_distances[t]`` contains one distance-vector for every prior
    decision i < t.  Thus its length must be t.  This directly represents the
    inner sum over conditional PMI terms in UProp Eq. 9.
    """
    iu = _finite_vector(intrinsic_uncertainties, "intrinsic_uncertainties")
    if np.any(iu < 0.0):
        raise ValueError("intrinsic uncertainties must be non-negative")
    if len(predecessor_distances) != len(iu):
        raise ValueError("one predecessor-distance list is required per step")

    extrinsic = []
    for t, per_prior in enumerate(predecessor_distances):
        if len(per_prior) != t:
            raise ValueError(
                f"step {t} must contain {t} prior-decision distance vectors"
            )
        extrinsic.append(sum(uprop_pmi(d, kernel=kernel) for d in per_prior))
    eu = np.asarray(extrinsic, dtype=float)
    total = float(np.sum(iu + eu))
    if not length_normalize:
        return total

    # UProp Eq. 9: lambda_z = T_z + sum_t EU_t / IU_t.  Zero/zero contributes
    # zero; positive EU with zero IU is stabilized by EPS.
    ratios = np.divide(eu, np.maximum(iu, EPS))
    lam = len(iu) + float(ratios.sum())
    return total / max(lam, EPS)


def uprop_score(
    tdps: Sequence[Mapping[str, object]],
    *,
    length_normalize: bool = True,
    kernel: Callable[[float, int], float] = uprop_kernel,
) -> float:
    """Average UProp score over TDP samples (UProp Eq. 9).

    Each mapping must contain ``intrinsic_uncertainties`` and
    ``predecessor_distances`` accepted by :func:`uprop_tdp_score`.
    """
    if not tdps:
        raise ValueError("at least one TDP sample is required")
    values = [
        uprop_tdp_score(
            tdp["intrinsic_uncertainties"],  # type: ignore[arg-type]
            tdp["predecessor_distances"],  # type: ignore[arg-type]
            length_normalize=length_normalize,
            kernel=kernel,
        )
        for tdp in tdps
    ]
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# HTC (Holistic Trajectory Calibration), 48 published trajectory features
# ---------------------------------------------------------------------------
HTC_FEATURE_NAMES = (
    "top1_gradient_mean", "top1_gradient_std", "top1_gradient_max",
    "top1_gradient_min", "top1_gradient_trend", "topk_gradient_mean",
    "topk_gradient_std", "topk_gradient_max", "topk_gradient_min",
    "topk_gradient_trend", "token_gradient_mean", "token_gradient_std",
    "token_gradient_max", "token_gradient_min", "step_progression_entropy",
    "step_progression_concentration", "step_progression_spread",
    "top1_confidence_change", "topk_confidence_change",
    "first_attention_entropy", "first_attention_concentration",
    "first_attention_spread", "first_confidence_volatility",
    "first_confidence_skewness", "first_top1_avg", "first_topk_avg",
    "last_attention_entropy", "last_attention_concentration",
    "last_attention_spread", "last_confidence_volatility",
    "last_confidence_skewness", "last_top1_avg", "last_topk_avg",
    "attention_entropy_mean", "attention_entropy_std",
    "attention_concentration_mean", "attention_concentration_std",
    "attention_spread_mean", "attention_spread_std",
    "token_volatility_mean", "token_volatility_std",
    "token_skewness_mean", "token_skewness_std", "normalized_step_count",
    "first_token_count", "last_token_count", "avg_tokens_per_step",
    "std_tokens_per_step",
)


def htc_features(steps: Sequence[Mapping[str, Sequence[float]]]) -> dict[str, float]:
    """Extract HTC's 48 features from a genuine multi-step trajectory.

    Every step must provide:

    - ``token_confidences``: positive confidence of each emitted token r_t,i;
    - ``top1_confidences``: top-1 vocabulary confidence at each token;
    - ``topk_confidences``: top-k vocabulary probability mass at each token.

    Undefined statistics are zero, as specified in HTC Appendix D.1.
    """
    if not steps:
        raise ValueError("HTC requires at least one trajectory step")

    summaries = []
    token_diffs: list[float] = []
    counts = []
    for index, step in enumerate(steps):
        r = _positive_vector(step.get("token_confidences", ()),
                             f"steps[{index}].token_confidences")
        top1 = _probability_vector(step.get("top1_confidences", ()),
                                   f"steps[{index}].top1_confidences")
        topk = _probability_vector(step.get("topk_confidences", ()),
                                   f"steps[{index}].topk_confidences")
        if not (len(r) == len(top1) == len(topk)):
            raise ValueError(f"all confidence arrays in step {index} must align")
        mu = float(r.mean())
        sigma = float(r.std())
        pi = r / (float(r.sum()) + 1e-8)
        entropy = float(-np.sum(pi * np.log(pi + 1e-8)))
        concentration = float(r.max() / (mu + 1e-8))
        spread = float(sigma / (mu + 1e-8))
        skew = float(np.mean(((r - mu) / (sigma + 1e-8)) ** 3))
        summaries.append({
            "entropy": entropy,
            "concentration": concentration,
            "spread": spread,
            "skew": skew,
            "top1": float(top1.mean()),
            "topk": float(topk.mean()),
        })
        token_diffs.extend(np.diff(r).tolist())
        counts.append(len(r))

    top1_values = np.asarray([s["top1"] for s in summaries])
    topk_values = np.asarray([s["topk"] for s in summaries])
    top1_grad = np.diff(top1_values)
    topk_grad = np.diff(topk_values)
    entropies = np.asarray([s["entropy"] for s in summaries])
    concentrations = np.asarray([s["concentration"] for s in summaries])
    spreads = np.asarray([s["spread"] for s in summaries])
    skews = np.asarray([s["skew"] for s in summaries])
    token_grad = np.asarray(token_diffs)
    counts_array = np.asarray(counts, dtype=float)

    features: dict[str, float] = {}
    _put_gradient_features(features, "top1", top1_grad, trend=True)
    _put_gradient_features(features, "topk", topk_grad, trend=True)
    _put_gradient_features(features, "token", token_grad, trend=False)
    features.update({
        "step_progression_entropy": _coefficient_of_variation(entropies),
        "step_progression_concentration": _coefficient_of_variation(concentrations),
        "step_progression_spread": _coefficient_of_variation(spreads),
        "top1_confidence_change": _change(top1_values),
        "topk_confidence_change": _change(topk_values),
    })
    _put_position_features(features, "first", summaries[0])
    _put_position_features(features, "last", summaries[-1])
    features.update({
        "attention_entropy_mean": float(entropies.mean()),
        "attention_entropy_std": float(entropies.std()),
        "attention_concentration_mean": float(concentrations.mean()),
        "attention_concentration_std": float(concentrations.std()),
        "attention_spread_mean": float(spreads.mean()),
        "attention_spread_std": float(spreads.std()),
        "token_volatility_mean": float(spreads.mean()),
        "token_volatility_std": float(spreads.std()),
        "token_skewness_mean": float(skews.mean()),
        "token_skewness_std": float(skews.std()),
        "normalized_step_count": len(steps) / 10.0,
        "first_token_count": float(counts_array[0]),
        "last_token_count": float(counts_array[-1]),
        "avg_tokens_per_step": float(counts_array.mean()),
        "std_tokens_per_step": float(counts_array.std()),
    })
    if tuple(features) != HTC_FEATURE_NAMES:
        raise AssertionError("internal HTC feature ordering/count mismatch")
    return features


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values")
    return x


def _positive_vector(values: Sequence[float], name: str) -> np.ndarray:
    x = _finite_vector(values, name)
    if np.any(x <= 0.0):
        raise ValueError(f"{name} must be positive")
    return x


def _probability_vector(values: Sequence[float], name: str) -> np.ndarray:
    x = _positive_vector(values, name)
    # Top-k probability mass is accumulated in model dtype (often float32 or
    # bfloat16), so a mathematically valid sum can land a few ulps above one.
    # Accept only that numerical overshoot; material invalid values still fail.
    if np.any(x > 1.0 + 1e-6):
        raise ValueError(f"{name} must be at most one")
    return np.minimum(x, 1.0)


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (float(values.mean()), float(values.std()),
            float(values.max()), float(values.min()))


def _put_gradient_features(out, prefix, values, *, trend):
    mean, std, maximum, minimum = _stats(values)
    out[f"{prefix}_gradient_mean"] = mean
    out[f"{prefix}_gradient_std"] = std
    out[f"{prefix}_gradient_max"] = maximum
    out[f"{prefix}_gradient_min"] = minimum
    if trend:
        out[f"{prefix}_gradient_trend"] = (
            float(values[-1] - values[0]) if values.size >= 2 else 0.0
        )


def _coefficient_of_variation(values: np.ndarray) -> float:
    return float(values.std() / (values.mean() + 1e-8))


def _change(values: np.ndarray) -> float:
    return float(values[-1] - values[0]) if values.size >= 2 else 0.0


def _put_position_features(out, prefix, summary):
    out[f"{prefix}_attention_entropy"] = summary["entropy"]
    out[f"{prefix}_attention_concentration"] = summary["concentration"]
    out[f"{prefix}_attention_spread"] = summary["spread"]
    out[f"{prefix}_confidence_volatility"] = summary["spread"]
    out[f"{prefix}_confidence_skewness"] = summary["skew"]
    out[f"{prefix}_top1_avg"] = summary["top1"]
    out[f"{prefix}_topk_avg"] = summary["topk"]
