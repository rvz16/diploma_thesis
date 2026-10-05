"""C6: adapt BFCL v3 (simple / multiple) into our constrained-decoding schema.

Loads the locally-cached BFCL dataset, converts each task into
  (tools spec for json_grammar, prompt, gold acceptable-answers),
filtering out tasks whose GOLD function needs argument types we don't support
yet (array / dict / nested objects). Correctness follows BFCL: a value is right
if it is in that argument's list of acceptable values.
"""

from __future__ import annotations

import glob
import json
import os

def _find_bfcl_dir():
    """Locate the cached BFCL data without crashing at import time.

    ``BFCL_DATA_DIR`` is useful on remote workers, where the Hugging Face cache
    is commonly mounted at a non-default path.  It must point at the directory
    containing files such as ``BFCL_v3_multiple.json``.
    """
    configured = os.environ.get("BFCL_DATA_DIR")
    if configured:
        return os.path.expanduser(configured)
    # A pinned project-local copy is used by ClearML workers.  It avoids an
    # implicit dependency on whichever Hugging Face cache happens to be mounted
    # on the remote machine.
    bundled = os.path.join(os.path.dirname(__file__), "data")
    if os.path.isdir(bundled):
        return bundled
    matches = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--gorilla-llm--berkeley-function-calling"
        "-leaderboard/snapshots/*"))
    return matches[0] if matches else None


BFCL_DIR = _find_bfcl_dir()

SUPPORTED = {"integer", "float", "number", "boolean", "string"}


def _load(name):
    if BFCL_DIR is None:
        raise FileNotFoundError(
            "BFCL data cache was not found. Set BFCL_DATA_DIR to the directory "
            "containing BFCL_v3_*.json, or download/pin the official BFCL data "
            "before running an evaluation."
        )
    candidates = [os.path.join(BFCL_DIR, name)]
    candidates.extend(glob.glob(os.path.join(BFCL_DIR, "bfcl_*", name)))
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing BFCL data file {name} below configured directory: {BFCL_DIR}"
        )
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _valuespec(prop):
    """Map a BFCL parameter property -> our json_grammar valuespec, or None if
    the type is unsupported."""
    t = prop.get("type")
    if "enum" in prop and prop["enum"]:
        return {"type": "enum", "values": [str(v) for v in prop["enum"]]}
    if t in ("integer",):
        return {"type": "integer"}
    if t in ("float", "number"):
        return {"type": "number"}
    if t == "boolean":
        return {"type": "boolean"}
    if t == "string":
        return {"type": "string"}
    return None                                    # array / dict / tuple / ...


def _func_to_spec(func):
    """-> (name, params) with params = [(key, valuespec, required), ...], or
    None if any REQUIRED param is an unsupported type. Optional params of
    unsupported type are simply dropped (omitting them is allowed)."""
    props = func["parameters"].get("properties", {})
    required = set(func["parameters"].get("required", []))
    params = []
    for k in props:                       # preserve declaration order
        vs = _valuespec(props[k])
        req = k in required
        if vs is None:
            if req:
                return None               # required + unsupported -> skip task
            continue                      # optional + unsupported -> drop it
        params.append((k, vs, req))
    return (func["name"], params)


def _describe(func):
    props = func["parameters"].get("properties", {})
    required = set(func["parameters"].get("required", []))
    parts = []
    for k, p in props.items():
        d = p.get("type", "any")
        if "enum" in p and p["enum"]:
            d = "enum[" + ",".join(map(str, p["enum"])) + "]"
        parts.append(f"{k}:{d}" + ("" if k in required else "?"))
    return f"- {func['name']}({', '.join(parts)}): {func.get('description','')[:80]}"


FORMAT_HINT = (
    ' Respond with ONLY a JSON object {"name": "<tool>", "arguments": {...}} '
    'and nothing else. Include exactly the required arguments.')


def build_bfcl(category="simple", limit=None, version="v3"):
    """Build a compatible one-step BFCL slice from a pinned data version."""
    if version not in {"v3", "v4"}:
        raise ValueError("version must be 'v3' or 'v4'")
    filename = f"BFCL_{version}_{category}.json"
    data = _load(filename)
    ans = {a["id"]: a for a in _load(f"possible_answer/{filename}")}
    tasks = []
    for e in data:
        funcs = e["function"]
        gt = ans[e["id"]]["ground_truth"]          # list of {fname: {arg:[...]}}
        gold_name = list(gt[0].keys())[0]
        gold_func = next((f for f in funcs if f["name"] == gold_name), None)
        if gold_func is None:
            continue
        # gold function must be fully representable
        if _func_to_spec(gold_func) is None:
            continue
        tools, ok = [], True
        for f in funcs:
            spec = _func_to_spec(f)
            if spec is None:
                # distractor with unsupported required args: give it free-string
                # args so it stays selectable but we don't crash
                spec = (f["name"], [(k, {"type": "string"}, True)
                                    for k in f["parameters"].get("required", [])])
            tools.append(spec)
        user_q = e["question"][0][0]["content"]
        desc = "Available tools:\n" + "\n".join(_describe(f) for f in funcs)
        tasks.append({
            "id": e["id"],
            "system": "You are a function-calling assistant." + FORMAT_HINT,
            "user": desc + "\n\nRequest: " + user_q,
            "tools": tools,
            "gold_name": gold_name,
            "gold_args": gt[0][gold_name],          # {arg: [acceptable values]}
        })
        if limit and len(tasks) >= limit:
            break
    return tasks


def is_correct(parsed, task):
    """BFCL-style: right function and every gold arg value in its acceptable
    list (string-normalized); no unexpected args."""
    if not isinstance(parsed, dict) or parsed.get("name") != task["gold_name"]:
        return False
    args = parsed.get("arguments", {})
    if not isinstance(args, dict):
        return False
    gold = task["gold_args"]
    for k, acceptable in gold.items():
        acc = {str(a).strip().lower() for a in acceptable}
        got = str(args.get(k, "")).strip().lower()
        if got not in acc:
            return False
    # any emitted arg not in gold (and non-empty) is a violation
    for k, v in args.items():
        if k not in gold and str(v).strip():
            return False
    return True


if __name__ == "__main__":
    for cat in ["simple", "multiple"]:
        ts = build_bfcl(cat)
        raw = len(_load(f"BFCL_v3_{cat}.json"))
        print(f"{cat}: {len(ts)}/{raw} tasks representable (supported types)")
    ex = build_bfcl("simple", limit=2)
    for t in ex:
        print("----", t["id"], "| gold:", t["gold_name"], t["gold_args"])
        print("   tools:", [(n, [a for a, _ in p]) for n, p in t["tools"]])
