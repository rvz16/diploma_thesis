"""
Constraint-aware function-call decoder with projection-pressure logging.

Decodes function calls of the form

    {"name": "<tool>", "arguments": {"<arg>": <value>, ...}}

under a hard, *branching* schema grammar: choosing the function name selects
that tool's argument sub-grammar. Argument values are restricted to
enum / integer / boolean (no free strings), so every emitted call is
schema-valid by construction and correctness is exact-match checkable.

Core instrumentation: the grammar is over *characters* but masking is applied
over *tokens*, allowing tokens to cross grammar boundaries (token healing). At
each step we record, BEFORE masking, the model's natural probability mass on the
schema-allowed token set A_t:

    M_t  = sum_{v in A_t} p_theta(v | y_<t, x)      (allowed mass)
    CP_t = 1 - M_t                                   (constraint pressure)
    PC   = sum_t -log M_t                            (projection cost)

We also store p of the actually-chosen token so classic UQ baselines (G-NLL,
MAX/AVG token uncertainty) are computable on the same emitted sequence. Each
step carries a role tag: syntax | function | argname | argvalue.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA must win over MPS/CPU: remote ClearML workers expose an NVIDIA GPU,
# while the original prototype was developed on Apple Silicon.
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")
END = "END"


def extract_json(text: str):
    """Best-effort: pull the first balanced {...} object out of free text and
    parse it. Returns a dict or None (schema-invalid at the parse level)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- #
# Branching grammar: a DAG of nodes; choosing a CHOICE option selects a next.
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("kind", "options", "role", "nexts")

    def __init__(self, kind, options, role):
        self.kind = kind                 # 'fixed' | 'choice'
        self.options = options           # fixed -> [text]; choice -> options
        self.role = role
        self.nexts: list[Node | None] = [None] * len(options)


def link(a: Node, b: Node | None):
    """Set every option of `a` to continue to `b`."""
    a.nexts = [b] * len(a.options)


def chain(nodes: list[Node]) -> Node:
    """Link a linear list of nodes; last continues to END (None)."""
    for i in range(len(nodes) - 1):
        link(nodes[i], nodes[i + 1])
    link(nodes[-1], None)
    return nodes[0]


class Grammar:
    def __init__(self, head: Node):
        self.head = head

    def start(self) -> frozenset:
        return self._closure({(self.head, j, 0)
                              for j in range(len(self.head.options))})

    def _closure(self, cursors) -> frozenset:
        out, stack = set(), list(cursors)
        while stack:
            c = stack.pop()
            if c == END:
                out.add(END)
                continue
            node, j, p = c
            if p == len(node.options[j]):           # finished this option
                nxt = node.nexts[j]
                if nxt is None:
                    out.add(END)
                else:
                    for k in range(len(nxt.options)):
                        stack.append((nxt, k, 0))
            else:
                out.add(c)
        return frozenset(out)

    def step(self, cursors, ch: str) -> frozenset:
        nxt = set()
        for c in cursors:
            if c == END:
                continue
            node, j, p = c
            opt = node.options[j]
            if p < len(opt) and opt[p] == ch:
                nxt.add((node, j, p + 1))
        return self._closure(nxt)

    @staticmethod
    def active_role(cursors) -> str:
        live = [c for c in cursors if c != END]
        if not live:
            return "syntax"
        roles = [c[0].role for c in live]
        # prefer informative roles over syntax when a token spans a boundary
        for pref in ("function", "argvalue", "argname"):
            if pref in roles:
                return pref
        return "syntax"


def build_fc_grammar(tools_spec: list[tuple[str, list[tuple[str, list[str]]]]]) -> Grammar:
    """tools_spec: [(tool_name, [(arg_name, [value_literals]), ...]), ...].
    value_literals are JSON literals: '"celsius"', '5', 'true'."""
    head = Node("fixed", ['{"name": "'], "syntax")
    fchoice = Node("choice", [t[0] for t in tools_spec], "function")
    link(head, fchoice)
    for j, (_name, args) in enumerate(tools_spec):
        nodes = [Node("fixed", ['", "arguments": {'], "syntax")]
        if not args:
            nodes.append(Node("fixed", ["}}"], "syntax"))
        else:
            for i, (arg, vals) in enumerate(args):
                sep = ", " if i > 0 else ""
                nodes.append(Node("fixed", [f'{sep}"{arg}": '], "argname"))
                nodes.append(Node("choice", list(vals), "argvalue"))
            nodes.append(Node("fixed", ["}}"], "syntax"))
        fchoice.nexts[j] = chain(nodes)
    return Grammar(head)


# --------------------------------------------------------------------------- #
@dataclass
class Step:
    role: str
    token_str: str
    m_t: float
    p_chosen: float
    # Unmasked distribution summaries used by the HTC trajectory baseline.
    p_top1: float = 0.0
    p_top5: float = 0.0
    slot: str | None = None      # argument name, for per-slot localization


@dataclass
class DecodeResult:
    text: str
    parsed: dict[str, Any] | None
    steps: list[Step] = field(default_factory=list)


class ConstrainedFC:
    MAXLEN = 32
    MAX_STR = 48        # max chars in a free-string value (prevents runaway loops)

    def __init__(self, model_name: str, id2str_cache: str | None = None,
                 dtype=torch.float32, local_files_only: bool = False):
        self.tok = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, local_files_only=local_files_only).to(DEVICE)
        self.model.eval()
        self.id2str = self._build_id2str(model_name, id2str_cache)
        self.str2ids: dict[str, list[int]] = {}
        for i, s in enumerate(self.id2str):
            if s:
                self.str2ids.setdefault(s, []).append(i)

    def _build_id2str(self, model_name, cache):
        if cache and os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)
        id2str = self.tok.batch_decode([[i] for i in range(len(self.tok))])
        if cache:
            with open(cache, "wb") as f:
                pickle.dump(id2str, f)
        return id2str

    def build_prefix(self, system, user) -> torch.Tensor:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").input_ids.to(DEVICE)

    # ---- unconstrained (prompt-only) generation, for the B3 constraint-tax
    #      arm. Same prompt, no grammar mask, greedy for a fair comparison. --- #
    def generate_free(self, system, user, max_new_tokens=160) -> DecodeResult:
        ids = self.build_prefix(system, user)
        with torch.no_grad():
            out = self.model.generate(
                ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        gen = out[0, ids.shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        parsed = extract_json(text)
        return DecodeResult(text=text, parsed=parsed, steps=[])

    def sample_free(self, system, user, n=8, temperature=0.7,
                    max_new_tokens=160):
        """N unconstrained samples (temperature) for multi-sample UQ baselines
        (AST-cluster / semantic entropy). Returns list of parsed dicts / None."""
        ids = self.build_prefix(system, user)
        with torch.no_grad():
            out = self.model.generate(
                ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.95,
                num_return_sequences=n, pad_token_id=self.tok.eos_token_id)
        outs = []
        for row in out:
            text = self.tok.decode(row[ids.shape[1]:], skip_special_tokens=True)
            outs.append(extract_json(text))
        return outs

    def sample_free_with_hidden(self, system, user, n=8, temperature=0.7,
                                max_new_tokens=160, layer=None, seed=None):
        """Sample calls and return their middle-layer sentence embeddings.

        This supports the INSIDE/EigenScore baseline.  INSIDE represents each
        sampled response by the hidden state of its last generated token in the
        middle transformer layer.  We generate first, then teacher-force the
        sampled sequences in one batch so that variable-length/EOS handling is
        explicit and the selected state really belongs to the last content
        token (rather than to padding).

        Returns ``(parsed_samples, last_embeddings, mean_embeddings, lengths)``.
        ``mean_embeddings`` is a documented ablation; the paper's main method
        uses ``last_embeddings``.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if DEVICE == "mps":
                torch.mps.manual_seed(seed)

        prefix = self.build_prefix(system, user)
        with torch.no_grad():
            generated = self.model.generate(
                prefix, attention_mask=torch.ones_like(prefix),
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.95,
                num_return_sequences=n, pad_token_id=self.tok.pad_token_id,
                return_dict_in_generate=True).sequences

        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_ids = set(eos_ids or [])
        if self.tok.eos_token_id is not None:
            eos_ids.add(self.tok.eos_token_id)

        prefix_len = prefix.shape[1]
        content_ids = []
        parsed_samples = []
        for row in generated:
            ids = row[prefix_len:].tolist()
            stop = next((i for i, tid in enumerate(ids) if tid in eos_ids),
                        len(ids))
            ids = ids[:stop]
            if not ids:                 # pathological immediate-EOS sample
                ids = row[prefix_len:prefix_len + 1].tolist()
            content_ids.append(ids)
            text = self.tok.decode(ids, skip_special_tokens=True)
            parsed_samples.append(extract_json(text))

        lengths = [len(ids) for ids in content_ids]
        full_lengths = [prefix_len + length for length in lengths]
        max_len = max(full_lengths)
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        batch = torch.full((n, max_len), pad_id, dtype=torch.long,
                           device=DEVICE)
        mask = torch.zeros((n, max_len), dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(content_ids):
            full = torch.cat((prefix[0], torch.tensor(ids, device=DEVICE)))
            batch[i, :len(full)] = full
            mask[i, :len(full)] = 1

        with torch.no_grad():
            out = self.model(input_ids=batch, attention_mask=mask,
                             output_hidden_states=True, use_cache=False)
        if layer is None:
            # hidden_states[0] is the embedding output; this matches the
            # official INSIDE choice int(len(hidden_states) / 2).
            layer = len(out.hidden_states) // 2
        hidden = out.hidden_states[layer].float()
        last_embeddings, mean_embeddings = [], []
        for i, length in enumerate(lengths):
            start, end = prefix_len, prefix_len + length
            last_embeddings.append(hidden[i, end - 1].detach().cpu())
            mean_embeddings.append(hidden[i, start:end].mean(0).detach().cpu())

        return (parsed_samples, torch.stack(last_embeddings),
                torch.stack(mean_embeddings), lengths)

    def _allowed_tokens(self, g: Grammar, cursors):
        """{token_id: (token_str, resulting_cursors)} for all vocab tokens whose
        string is a grammar-valid continuation (may cross boundaries / reach END)."""
        allowed: dict[int, tuple[str, frozenset]] = {}
        frontier = [("", cursors)]
        while frontier:
            new_frontier = []
            for pre, cur in frontier:
                chars = set()
                for c in cur:
                    if c == END:
                        continue
                    node, j, p = c
                    opt = node.options[j]
                    if p < len(opt):
                        chars.add(opt[p])
                for ch in chars:
                    npre = pre + ch
                    ncur = g.step(cur, ch)
                    if not ncur:
                        continue
                    for tid in self.str2ids.get(npre, []):
                        allowed[tid] = (npre, ncur)
                    if len(npre) < self.MAXLEN and any(c != END for c in ncur):
                        new_frontier.append((npre, ncur))
            frontier = new_frontier
        return allowed

    def generate_call(self, system, user, tools_spec) -> DecodeResult:
        g = build_fc_grammar(tools_spec)
        ids = self.build_prefix(system, user)
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[0, -1]

        cursors = g.start()
        steps: list[Step] = []
        gen_ids: list[int] = []

        while True:
            if all(c == END for c in cursors):
                break
            allowed = self._allowed_tokens(g, cursors)
            if not allowed:
                break
            probs = torch.softmax(last_logits.float(), dim=-1)
            top = torch.topk(probs, k=min(5, len(probs))).values
            p_top1, p_top5 = float(top[0].item()), float(top.sum().item())
            tid_list = list(allowed.keys())
            ids_t = torch.tensor(tid_list, device=probs.device)
            m_t = float(probs[ids_t].sum().item())
            best = int(torch.argmax(last_logits[ids_t]).item())
            chosen_id = tid_list[best]
            p_chosen = float(probs[chosen_id].item())
            tok_str, ncur = allowed[chosen_id]

            steps.append(Step(role=g.active_role(cursors), token_str=tok_str,
                              m_t=m_t, p_chosen=p_chosen,
                              p_top1=p_top1, p_top5=p_top5))
            gen_ids.append(chosen_id)
            cursors = ncur

            inp = torch.tensor([[chosen_id]], device=DEVICE)
            with torch.no_grad():
                out = self.model(input_ids=inp, past_key_values=past,
                                 use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[0, -1]
            if len(gen_ids) > 128:
                break

        text = self.tok.decode(gen_ids)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        return DecodeResult(text=text, parsed=parsed, steps=steps)

    # ------------------------------------------------------------------ #
    # C5: decode against a general JSON-schema NFA (json_grammar.py)
    # ------------------------------------------------------------------ #
    def _ensure_json_index(self):
        if hasattr(self, "_safe_ids"):
            return
        from json_grammar import SAFE
        safe, quote, first = [], [], {}
        for tid, s in enumerate(self.id2str):
            if not s:
                continue
            first.setdefault(s[0], []).append(tid)
            if '"' in s:
                quote.append(tid)
            elif all(c in SAFE for c in s):
                safe.append(tid)
        self._safe_ids = torch.tensor(safe, device=DEVICE)
        self._quote_ids = quote
        self._first_char_map = first

    def _nfa_consume(self, states, s):
        import json_grammar as jg
        cur = states
        for ch in s:
            cur = jg.step(cur, ch)
            if not cur:
                return None
        return cur

    def _nfa_allowed(self, states):
        """Return (allowed dict tid->(str,result_states), in_string_body)."""
        import json_grammar as jg
        if any(st.is_str for st in states):
            allowed = {}
            for tid in self._quote_ids:
                s = self.id2str[tid]
                res = self._nfa_consume(states, s)
                if res:
                    allowed[tid] = (s, res)
            return allowed, True
        firsts = set()
        for st in states:
            for cs, _ in st.trans:
                firsts |= cs
        allowed, seen = {}, set()
        for c in firsts:
            for tid in self._first_char_map.get(c, ()):
                if tid in seen:
                    continue
                seen.add(tid)
                s = self.id2str[tid]
                res = self._nfa_consume(states, s)
                if res:
                    allowed[tid] = (s, res)
        return allowed, False

    def generate_schema(self, system, user, tools) -> DecodeResult:
        """tools: list of (func_name, params); params: [(key, valuespec), ...]
        valuespec: {'type': 'enum'|'string'|'integer'|'number'|'boolean',
                    'values': [...]}  (values only for enum)."""
        import json_grammar as jg
        self._ensure_json_index()
        head = jg.compile_call(tools)
        states = jg.start_states(head)

        ids = self.build_prefix(system, user)
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[0, -1]
        steps, gen_ids = [], []
        str_len = 0                     # content chars in the current free string

        while True:
            if jg.is_accepting(states):
                break
            allowed, in_str = self._nfa_allowed(states)
            probs = torch.softmax(last_logits.float(), dim=-1)
            top = torch.topk(probs, k=min(5, len(probs))).values
            p_top1, p_top5 = float(top[0].item()), float(top.sum().item())
            role = jg.role_of(states)

            if in_str:
                q_ids = list(allowed.keys())
                q_t = torch.tensor(q_ids, device=DEVICE) if q_ids else None
                # bound string length: greedy repetition can otherwise never
                # emit the (always-allowed) closing quote -> force-close.
                if str_len >= self.MAX_STR and q_t is not None:
                    m_t = float(probs[q_t].sum().item())
                    chosen_id = int(q_t[torch.argmax(last_logits[q_t])].item())
                    ncur = allowed[chosen_id][1]
                else:
                    mass = float(probs[self._safe_ids].sum().item())
                    if q_t is not None:
                        mass += float(probs[q_t].sum().item())
                    bs = int(self._safe_ids[torch.argmax(
                        last_logits[self._safe_ids])].item())
                    best_id, best_logit, best_res = bs, float(last_logits[bs]), \
                        states
                    if q_t is not None:
                        bq = int(q_t[torch.argmax(last_logits[q_t])].item())
                        if float(last_logits[bq]) > best_logit:
                            best_id, best_res = bq, allowed[bq][1]
                    chosen_id, ncur, m_t = best_id, best_res, mass
            else:
                if not allowed:
                    break
                tid_list = list(allowed.keys())
                ids_t = torch.tensor(tid_list, device=DEVICE)
                m_t = float(probs[ids_t].sum().item())
                chosen_id = tid_list[int(torch.argmax(last_logits[ids_t]).item())]
                ncur = allowed[chosen_id][1]

            p_chosen = float(probs[chosen_id].item())
            steps.append(Step(role=role, token_str=self.id2str[chosen_id],
                              m_t=m_t, p_chosen=p_chosen,
                              p_top1=p_top1, p_top5=p_top5,
                              slot=jg.active_slot(states)))
            gen_ids.append(chosen_id)
            states = ncur
            # track free-string content length (reset once the string closes)
            str_len = str_len + len(self.id2str[chosen_id]) \
                if any(st.is_str for st in ncur) else 0

            inp = torch.tensor([[chosen_id]], device=DEVICE)
            with torch.no_grad():
                out = self.model(input_ids=inp, past_key_values=past,
                                 use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[0, -1]
            if len(gen_ids) > 160:
                break

        text = self.tok.decode(gen_ids)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        return DecodeResult(text=text, parsed=parsed, steps=steps)
