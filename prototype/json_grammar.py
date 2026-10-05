"""C5: a character-level NFA for real-ish JSON function-call schemas.

Extends the fixed-string grammar to the value types BFCL needs:
  enum(string) | string(free) | integer | number | boolean, with required args.

Design notes
------------
* The grammar is a small NFA over characters (states + char-set edges + epsilon
  edges). Token masking is applied on top (token healing), and every state
  carries a role tag (function / argname / argvalue / syntax) so we keep the
  role-decomposed constraint-pressure signal that готовые libs don't expose.
* Free strings are the only "wide" construct. We handle them without character
  BFS explosion: inside a string body, allowed tokens = precomputed
  string-safe tokens (stay) plus quote-containing tokens whose post-quote tail
  is grammar-valid (close / heal). String content carries ~0 constraint
  pressure by design, which matches the finding that the signal lives at
  function / enum / number / structural positions.
"""

from __future__ import annotations

# characters allowed inside a free JSON string (no quote, backslash, controls)
SAFE = frozenset(
    " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~")
DIGITS = frozenset("0123456789")


class State:
    __slots__ = ("role", "trans", "eps", "is_str", "str_next", "accept", "slot")

    def __init__(self, role="syntax"):
        self.role = role
        self.trans: list[tuple[frozenset, State]] = []
        self.eps: list[State] = []
        self.is_str = False
        self.str_next: State | None = None
        self.accept = False
        self.slot: str | None = None      # argument name (for localization)


# --------------------------------------------------------------------------- #
# NFA execution
# --------------------------------------------------------------------------- #
def closure(states):
    out, stack = set(), list(states)
    while stack:
        s = stack.pop()
        if s in out:
            continue
        out.add(s)
        for t in s.eps:
            stack.append(t)
    return frozenset(out)


def step(states, ch):
    nxt = set()
    for s in states:
        if s.is_str:
            if ch in SAFE:
                nxt.add(s)
            elif ch == '"':
                nxt.add(s.str_next)
        else:
            for cs, t in s.trans:
                if ch in cs:
                    nxt.add(t)
    return closure(nxt)


def role_of(states):
    roles = {s.role for s in states}
    for pref in ("function", "argvalue", "argname"):
        if pref in roles:
            return pref
    return "syntax"


def active_slot(states):
    for s in states:
        if s.slot is not None:
            return s.slot
    return None


def _tag_slot(entry, cont, slot):
    """Tag every state of a value fragment (from entry, stopping at cont)."""
    seen, stack = set(), [entry]
    while stack:
        s = stack.pop()
        if s is cont or s in seen:
            continue
        seen.add(s)
        s.slot = slot
        for _, t in s.trans:
            stack.append(t)
        stack.extend(s.eps)
        if s.is_str and s.str_next is not None:
            stack.append(s.str_next)


def is_accepting(states):
    return any(s.accept for s in states)


# --------------------------------------------------------------------------- #
# Schema -> NFA compiler (continuation-passing: build(cont) -> entry state)
# --------------------------------------------------------------------------- #
def lit(text, cont, role="syntax"):
    st = cont
    for ch in reversed(text):
        ns = State(role)
        ns.trans = [(frozenset({ch}), st)]
        st = ns
    return st


def free_string(cont, role="argvalue"):
    body = State(role)
    body.is_str = True
    body.str_next = cont
    return lit('"', body, role)          # opening quote -> string body


def _digit_chain(n, exit_state, role):
    """Entry state accepting 0..n digits then epsilon to exit_state (bounded, so
    greedy repetition can't run away)."""
    st = exit_state
    for _ in range(n):
        ns = State(role)
        ns.trans = [(DIGITS, st)]
        ns.eps = [exit_state]
        st = ns
    return st


def number(cont, role="argvalue", max_int=15, max_frac=15):
    """JSON number: -?(0|[1-9][0-9]*)(\\.[0-9]+)?  (no leading zeros, bounded)."""
    nonzero = frozenset("123456789")
    afterint = State(role)          # integer part complete
    fracfirst = State(role)
    afterint.trans = [(frozenset({'.'}), fracfirst)]
    afterint.eps = [cont]
    fracbody = _digit_chain(max_frac - 1, cont, role)   # after 1st frac digit
    fracfirst.trans = [(DIGITS, fracbody)]
    intbody = _digit_chain(max_int - 1, afterint, role)  # after leading nonzero
    firstdig = State(role)          # first integer digit (after '-')
    firstdig.trans = [(frozenset({'0'}), afterint), (nonzero, intbody)]
    start = State(role)
    start.trans = [(frozenset({'-'}), firstdig),
                   (frozenset({'0'}), afterint), (nonzero, intbody)]
    return start


def enum(values, cont, role="argvalue"):
    """Trie over the given literal strings (each incl. its quotes), -> cont."""
    root = State(role)
    for v in values:
        cur = root
        for i, ch in enumerate(v):
            last = i == len(v) - 1
            nxt = None
            for cs, t in cur.trans:
                if ch in cs:
                    nxt = t
                    break
            if nxt is None:
                nxt = cont if last else State(role)
                cur.trans.append((frozenset({ch}), nxt))
            elif last and cont not in nxt.eps:
                nxt.eps.append(cont)       # value is a prefix of another
            cur = nxt
    return root


def compile_value(spec, cont):
    t = spec["type"]
    if t == "enum":
        return enum([f'"{v}"' for v in spec["values"]], cont)
    if t == "string":
        return free_string(cont)
    if t == "boolean":
        return enum(["true", "false"], cont)
    if t in ("integer", "number"):
        return number(cont)
    # fallback: treat unknown types as a free string
    return free_string(cont)


def compile_object(params, cont):
    """params: list of (key, valuespec[, required]) in order. Required args are
    always emitted; optional args may be present or absent, with correct commas.

    Forward two-state construction: P[i] = considering entries[i:] with nothing
    emitted yet, Q[i] = considering entries[i:] with something already emitted.
    The first emitted entry gets no leading comma (from P), later ones do
    (from Q)."""
    params = [(p if len(p) == 3 else (p[0], p[1], True)) for p in params]
    n = len(params)
    close = lit("}", cont, "syntax")
    P = [None] * (n + 1)
    Q = [None] * (n + 1)
    P[n] = Q[n] = close
    for i in range(n - 1, -1, -1):
        key, vspec, req = params[i]
        val_p = compile_value(vspec, Q[i + 1])       # no leading comma
        _tag_slot(val_p, Q[i + 1], key)
        emit_p = lit(f'"{key}": ', val_p, "argname")
        val_q = compile_value(vspec, Q[i + 1])       # leading comma
        _tag_slot(val_q, Q[i + 1], key)
        emit_q = lit(", ", lit(f'"{key}": ', val_q, "argname"), "syntax")
        if req:
            P[i], Q[i] = emit_p, emit_q
        else:
            pi, qi = State("syntax"), State("syntax")
            pi.eps = [emit_p, P[i + 1]]              # emit or skip
            qi.eps = [emit_q, Q[i + 1]]
            P[i], Q[i] = pi, qi
    return lit("{", P[0], "syntax")


def compile_call(tools):
    """tools: list of (func_name, params). Returns the start State.
    Grammar: {"name": "<func>", "arguments": {<params>}}"""
    end = State("syntax")
    end.accept = True
    # per-function suffix: '", "arguments": ' + object + '}' -> end
    leaves = []
    for name, params in tools:
        obj = compile_object(params, lit("}", end, "syntax"))
        cont = lit('", "arguments": ', obj, "syntax")
        leaves.append((name, cont))
    # function-name trie (unquoted; surrounding quotes come from the literals)
    root = State("function")
    for name, cont in leaves:
        cur = root
        for i, ch in enumerate(name):
            last = i == len(name) - 1
            nxt = None
            for cs, t in cur.trans:
                if ch in cs:
                    nxt = t
                    break
            if nxt is None:
                nxt = cont if last else State("function")
                cur.trans.append((frozenset({ch}), nxt))
            elif last and cont not in nxt.eps:
                nxt.eps.append(cont)
            cur = nxt
    return lit('{"name": "', root, "syntax")


def start_states(head):
    return closure({head})
