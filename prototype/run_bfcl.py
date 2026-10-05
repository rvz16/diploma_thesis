"""C6: run the constrained-decoding + constraint-pressure pipeline on BFCL."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from bfcl_adapter import build_bfcl, is_correct
from fc_constrained import ConstrainedFC
from run_poc import features_from_steps


def main(category, model_name, out_path, limit=None, cache="id2str_qwen05.pkl",
         dtype=None, data_version="v3"):
    import torch
    tasks = build_bfcl(category, limit=limit, version=data_version)
    fc = ConstrainedFC(model_name, id2str_cache=cache,
                       dtype=dtype or torch.float32)
    rows, t0, n_corr, n_valid = [], time.time(), 0, 0
    for i, t in enumerate(tasks):
        r = fc.generate_schema(t["system"], t["user"], t["tools"])
        feats = features_from_steps(r.steps)
        feats["schema_complexity"] = len(t["tools"]) + sum(
            len(p) for _, p in t["tools"])
        corr = is_correct(r.parsed, t)
        n_corr += corr
        n_valid += r.parsed is not None
        rows.append({"id": t["id"], "family": category,
                     "gold_name": t["gold_name"], "pred": r.text,
                     "correct": int(corr), "wrong_valid": int(not corr),
                     "valid_json": int(r.parsed is not None), **feats})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tasks)} acc={n_corr/(i+1):.3f} "
                  f"{(time.time()-t0)/(i+1):.2f}s/ex", file=sys.stderr)
    n = len(rows)
    print(f"\nBFCL {data_version} {category}: n={n} acc={n_corr/n:.3f} wrong_valid={1-n_corr/n:.3f} "
          f"valid_json={n_valid}/{n} time={time.time()-t0:.0f}s")
    result = {"model": model_name, "bfcl_version": data_version, "rows": rows}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out_path)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", default="simple")
    p.add_argument("model", nargs="?", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("limit", nargs="?", type=int)
    p.add_argument("--output", help="Preserve existing artifacts by writing here")
    p.add_argument("--data-version", choices=("v3", "v4"), default="v3")
    p.add_argument("--bf16", action="store_true",
                   help="Load weights in bfloat16 (required for larger GPU models).")
    return p.parse_args()


if __name__ == "__main__":
    import torch
    args = parse_args()
    cat, model, lim = args.category, args.model, args.limit
    tag = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    cache = f"id2str_{tag}.pkl"
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    out = args.output or (f"features_bfcl_{cat}_{tag}.json"
                          if tag != "qwen_qwen2_5_0_5b_instruct"
                          else f"features_bfcl_{cat}.json")
    main(cat, model, out, lim, cache, dtype, args.data_version)
