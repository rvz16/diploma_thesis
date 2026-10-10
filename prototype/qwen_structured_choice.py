"""Matched-size autoregressive structured-choice backend for Clef comparisons.

Clef is post-trained from Qwen/Qwen3.8-27B.  This module loads that exact-size
base model and evaluates the same bounded BFCL candidate-choice records.  The
autoregressive output is constrained to one of four JSON objects::

    {"action":"A"}

For short A/B/C/D labels the tokenized valid outputs share a common prefix and
suffix and differ at exactly one token.  We force the common JSON syntax and
renormalize the model logits over the four allowed branch tokens.  This is the
exact greedy result for that finite structured-output language, with no parsing
failures and with a native probability distribution over the same classes that
Clef scores.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable


QWEN_MODEL = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def single_token_choice_branch(tokenizer: Any, labels: list[str]) -> tuple[list[int], dict[str, int]]:
    """Return the common JSON-token prefix and one unique branch token/label.

    Refuse to silently approximate a multi-token or label-dependent suffix.  A
    failed assertion means that the requested labels need the general trie
    decoder rather than this controlled single-decision experiment.
    """
    sequences = {
        label: tokenizer.encode(
            json.dumps({"action": label}, ensure_ascii=False, separators=(",", ":")),
            add_special_tokens=False,
        )
        for label in labels
    }
    if not sequences:
        raise ValueError("At least one label is required")
    rows = list(sequences.values())
    prefix_len = 0
    while all(prefix_len < len(row) for row in rows):
        tokens = {row[prefix_len] for row in rows}
        if len(tokens) != 1:
            break
        prefix_len += 1
    tails = {label: row[prefix_len:] for label, row in sequences.items()}
    if any(not tail for tail in tails.values()):
        raise ValueError("A label is a token-prefix of another valid structured output")
    branch = {label: tail[0] for label, tail in tails.items()}
    if len(set(branch.values())) != len(labels):
        raise ValueError("Labels do not separate at a unique next-token branch")
    suffixes = [tail[1:] for tail in tails.values()]
    if any(suffix != suffixes[0] for suffix in suffixes[1:]):
        raise ValueError("Labels have token-dependent suffixes; use a general trie decoder")
    return rows[0][:prefix_len], branch


def _prompt(body: dict[str, Any]) -> str:
    record = {
        "state": body["state"],
        "question": body["questions"]["action"],
    }
    return (
        "Select exactly one candidate action. All candidates are schema-valid. "
        "Return only a JSON object matching {\"action\": <allowed option ID>}.\n\n"
        + json.dumps(record, ensure_ascii=False, sort_keys=True)
    )


def load_qwen_structured_choice() -> tuple[
    Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]
]:
    """Load Qwen3.8-27B and expose a SystemOne-shaped decision callable."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_path = snapshot_download(QWEN_MODEL, revision=QWEN_REVISION)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device

    def decide(body: dict[str, Any]) -> dict[str, Any]:
        labels = list(body["questions"]["action"]["criteria"])
        common_prefix, branch = single_token_choice_branch(processor.tokenizer, labels)
        messages = [{"role": "user", "content": _prompt(body)}]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in encoded.items()}
        prefix = torch.tensor([common_prefix], dtype=inputs["input_ids"].dtype, device=device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
        if "attention_mask" in inputs:
            mask = torch.ones_like(prefix, dtype=inputs["attention_mask"].dtype)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], mask], dim=1)

        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
        latency_ms = (time.perf_counter() - started) * 1000
        option_logits = torch.stack([logits[branch[label]] for label in labels])
        option_probs = option_logits.softmax(dim=0).cpu().tolist()
        probabilities = dict(zip(labels, option_probs))
        choice = max(labels, key=probabilities.__getitem__)
        return {
            "model": QWEN_MODEL,
            "answers": {
                "action": {
                    "choice": choice,
                    "confidence": probabilities[choice],
                    "probabilities": probabilities,
                    "structured_output": json.dumps(
                        {"action": choice}, separators=(",", ":")
                    ),
                }
            },
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "decision_tokens": 1,
            },
            "latency_ms": latency_ms,
        }

    return decide, {
        "model": QWEN_MODEL,
        "api_model": QWEN_MODEL,
        "parameters": 27_000_000_000,
        "weights_revision": QWEN_REVISION,
        "precision": "bfloat16",
        "decoding": "finite JSON schema; single constrained enum-token branch",
        "thinking": False,
    }

