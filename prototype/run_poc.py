"""Generate constrained calls, compute UQ + constraint-pressure features, label
wrong-valid, and dump a feature table for evaluation."""

from __future__ import annotations

import json
import math
import sys
import time

from dataset_stress import build_dataset
from fc_constrained import ConstrainedFC

EPS = 1e-12
SEM_ROLES = {"function", "argvalue"}


def nlog(x):
    return -math.log(max(x, EPS))


def _agg(vals):
    """(sum, mean, max, top3-mean) for a list; zeros if empty."""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    top = sorted(vals, reverse=True)[:3]
    return (sum(vals), sum(vals) / len(vals), max(vals), sum(top) / len(top))


def features_from_steps(steps):
    p = [s.p_chosen for s in steps]
    m = [s.m_t for s in steps]
    sem = [i for i, s in enumerate(steps) if s.role in SEM_ROLES]
    fn = [i for i, s in enumerate(steps) if s.role == "function"]
    av = [i for i, s in enumerate(steps) if s.role == "argvalue"]
    tok_unc = [nlog(x) for x in p]
    cp = [1.0 - x for x in m]                     # per-token constraint pressure
    cp_av = [cp[i] for i in av]
    cp_sem = [cp[i] for i in sem]
    av_sum, av_mean, av_max, av_top3 = _agg(cp_av)
    sem_sum, sem_mean, sem_max, sem_top3 = _agg(cp_sem)
    nt = max(len(steps), 1)
    return {
        # ---- baselines (log-prob of the emitted tokens) ----
        "gnll": sum(tok_unc),
        "gnll_smt": sum(tok_unc[i] for i in sem),
        "avg_sem_tok_unc": (sum(tok_unc[i] for i in sem) / len(sem)
                            if sem else 0.0),
        "max_tok_unc": max(tok_unc) if tok_unc else 0.0,
        "avg_tok_unc": sum(tok_unc) / nt,
        # ---- constraint pressure: length-normalized / localized ----
        # (drop-in replacements for the saturated cp_max & length-confounded sums)
        "pc_mean": sum(nlog(x) for x in m) / nt,          # avg projection cost
        "cp_argvalue_mean": av_mean,                       # value-slot pressure
        "cp_argvalue_max": av_max,
        "cp_argvalue_top3": av_top3,
        "cp_sem_mean": sem_mean,
        "cp_sem_max": sem_max,
        "cp_sem_top3": sem_top3,
        "cp_function": sum(cp[i] for i in fn),
        # ---- kept for backward-compat / ablation (deprecated) ----
        "pc": sum(nlog(x) for x in m),
        "pc_sem": sum(nlog(m[i]) for i in sem),
        "cp_max": max(cp) if cp else 0.0,                  # saturated -> unused
        "cp_argvalue": av_sum,
        "n_tokens": len(steps),
        "n_sem_tokens": len(sem),
    }


def normval(v):
    return str(v).strip().lower()


def is_correct(parsed, gold):
    if parsed is None or "name" not in parsed:
        return False
    if parsed.get("name") != gold["name"]:
        return False
    args = parsed.get("arguments", {})
    for k, gv in gold["arguments"].items():
        if normval(args.get(k)) != normval(gv):
            return False
    return True


def schema_complexity(tools_spec):
    return len(tools_spec) + sum(len(a) for _, a in tools_spec)


def main(model_name, out_path, limit=None):
    fc = ConstrainedFC(model_name, id2str_cache="id2str_qwen05.pkl")
    ds = build_dataset()
    if limit:
        ds = ds[:limit]
    rows = []
    t0 = time.time()
    n_corr = 0
    for i, e in enumerate(ds):
        r = fc.generate_call(e["system"], e["user"], e["tools_spec"])
        feats = features_from_steps(r.steps)
        feats["schema_complexity"] = schema_complexity(e["tools_spec"])
        corr = is_correct(r.parsed, e["gold"])
        n_corr += corr
        rows.append({
            "id": e["id"], "family": e["family"],
            "request": e["user"].splitlines()[-1],
            "pred": r.text, "gold": e["gold"],
            "correct": int(corr), "wrong_valid": int(not corr),
            "valid_json": int(r.parsed is not None),
            **feats,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds)}  acc={n_corr/(i+1):.3f}  "
                  f"{(time.time()-t0)/(i+1):.2f}s/ex", file=sys.stderr)
    dt = time.time() - t0
    acc = n_corr / len(rows)
    print(f"\nmodel={model_name}  n={len(rows)}  acc={acc:.3f}  "
          f"wrong_valid_rate={1-acc:.3f}  valid_json="
          f"{sum(r['valid_json'] for r in rows)}/{len(rows)}  total={dt:.1f}s")
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "rows": rows}, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    out = sys.argv[2] if len(sys.argv) > 2 else "features_qwen05.json"
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(model, out, lim)
