# Prototype: Projection-Induced Constraint Pressure for Wrong-Valid FC Detection

A minimal end-to-end test of the thesis's central idea (`main_text.tex`):

> *Does the probability mass removed by constrained decoding reveal hidden
> semantic uncertainty and help detect wrong-valid function calls before
> execution?*

## What it does

1. **`fc_constrained.py`** — a self-contained, *branching* schema-constrained
   decoder for function calls `{"name": ..., "arguments": {...}}`. The grammar
   is over characters but masking is applied over tokens (with token healing, so
   tokens may cross grammar boundaries — this avoids fake tokenization
   pressure). At every step it logs the **allowed probability mass**
   `M_t = Σ_{v∈A_t} p_θ(v)` *before* masking, plus `p` of the emitted token, with
   a role tag (syntax / function / argname / argvalue).
   - `CP_t = 1 − M_t` (constraint pressure), `PC = Σ −log M_t` (projection cost).
2. **`dataset_stress.py`** — *Constraint-Stress-FC*: 116 controlled tasks across
   5 tool families with confusable tools + strict enum/int/bool args. Every
   constrained output is schema-valid, so `wrong_valid ≡ (not correct)` and
   labels are exact-match against gold.
3. **`run_poc.py`** — generates calls, computes UQ baselines (G-NLL, G-NLL-SMT,
   MAX/AVG token uncertainty) and CP features (PC, PC_sem, CP_max, CP_function,
   CP_argvalue), labels wrong-valid, dumps `features_*.json`.
4. **`evaluate.py`** — per-feature AUROC/AUPRC, cross-validated combined
   detectors, risk-coverage curves + Risk@80%cov (`*_riskcov.png`).
5. **`analyze.py`** — **confound control**: per-family rates + Leave-One-Family-Out
   CV (train on 4 families, test on a held-out 5th) to check whether CP is real
   per-call signal or a tool-family/length shortcut.

## How to run

```bash
python run_poc.py Qwen/Qwen2.5-0.5B-Instruct features_qwen05.json
python evaluate.py features_qwen05.json
python analyze.py  features_qwen05.json
```

Runs on Apple Silicon (MPS) / CPU; ~0.7 s/example, no GPU needed for the PoC.
Model: `Qwen2.5-0.5B-Instruct` (cached). `id2str_qwen05.pkl` caches the vocab
surface strings.

## Headline findings (Qwen2.5-0.5B, n=116, 26% wrong-valid)

| Feature set | Random-split AUROC | Cross-family AUROC (unseen families) |
|---|---|---|
| G-NLL-SMT (seed-paper baseline) | 0.52 | 0.49 (chance) |
| Logit baselines (G-NLL/MAX/AVG/SMT) | 0.61 | 0.23 (overfits to family) |
| **Constraint pressure (5 feats)** | **0.88** | **0.73** |
| Baselines + CP | 0.91 | 0.73 |
| Full detector | 0.93 | — |

- **No single feature works** (AUROC 0.42–0.62); the CP signal is multivariate.
- Adding CP halves Risk@80%cov (0.19 → 0.12 in-distribution).
- **The 0.88–0.91 in-distribution numbers are partly a tool-family confound.**
  After Leave-One-Family-Out, baselines collapse to chance/anti-predictive
  (0.23–0.49) but **CP still transfers at 0.73** — so constraint pressure
  captures generation-process uncertainty that is genuinely family-independent,
  not just "which family is hard."
- `cp_max` looks saturated (~0.99 everywhere) but its *small* variance
  (calendar 0.989 vs 0.999) is informative — see A2 note below.

### A2 result: feature engineering was tested and REJECTED

Attempts to "clean up" the CP feature set all **regressed the honest
(cross-family / Leave-One-Family-Out) metric**:

| CP feature set | Random AUROC | Cross-family (LOFO) |
|---|---|---|
| **Original (raw sums + cp_max)** | 0.88 | **0.73** |
| Length-normalized / localized | 0.76 | 0.64 |
| Drop cp_max + add cp_argvalue_mean | — | 0.52 |
| + baselines (normalized) | 0.86 | 0.30 |

Lesson: the "principled" normalized features transfer *worse*. Localized
arg-value pressure is great on value-error families (media 0.93, messaging
0.86) but fails on smart_home (0.14), where errors are *function*-selection,
not value errors — caught only by `pc`/`cp_function`. **Canonical set kept as
the original `[pc, pc_sem, cp_max, cp_function, cp_argvalue]`.** Verified by
`test_decoder.py` (8/8: M_t == brute-force allowed mass at every step).

This supports thesis hypotheses **H-A2** (CP differs for wrong-valid) and
**H-B1** (CP adds signal over G-NLL-SMT), with the honest caveat that the
in-distribution lift is inflated by family structure.

## B3 result: the constraint tax (H-A1)

`run_constraint_tax.py` — constrained vs prompt-only, same fair (explicit-format)
prompt, greedy, n=116:

| Arm | correct | wrong-valid | invalid |
|---|---|---|---|
| prompt-only | 0.48 | 0.17 | 0.34 |
| constrained | 0.76 | **0.24** | 0.00 |

Paired transition (free → constrained): of 40 prompt-only **invalids**, 70% are
rescued to correct but **30% become wrong-valid**; free wrong-valids stay wrong
80% of the time; free corrects stay correct 100%. **H-A1 supported**: hard
constraints raise schema validity to 100% and lift wrong-valid 0.17→0.24 — the
tax is a *loud→silent* error conversion, even though accuracy improves overall.
This is the core motivation: constraining manufactures the wrong-valid calls the
detector must catch.

**Confirmed on real BFCL** (`run_bfcl_tax.py`, clean grammar, Qwen-0.5B):

| Slice | arm | schema-valid | wrong-valid | net acc |
|---|---|---|---|---|
| simple | prompt-only | 0.80 | 0.30 | 0.50 |
| simple | constrained | 1.00 | **0.36** | 0.64 |
| live | prompt-only | 0.76 | 0.50 | 0.26 |
| live | constrained | 1.00 | **0.68** | 0.32 |

Same story on real data: schema-validity → 100%, accuracy improves, but
wrong-valid rises; prompt-only *invalid* errors split ~half rescued / half into
silent wrong-valid. (An earlier version wrongly showed constraining "corrupting"
34–42% of correct calls — that was the dropped-optional-arg artifact; after the
fix only 6% (simple) / 28% (live) of free-correct calls flip, mostly genuine.)

## C5+C6 result: real BFCL — the honest, tempered verdict

`json_grammar.py` (char-NFA: string/number/enum/bool, brute-force-verified M_t,
`test_json_grammar.py` 7/7) + `bfcl_adapter.py` + `run_bfcl.py`. Qwen-0.5B on
BFCL simple (n=347, 36% wrong-valid) and multiple (n=174, 45%), 100% valid JSON.

Wrong-valid detection AUROC (5-fold CV, **clean labels** after the optional-arg
fix below):

| Detector | 0.5B mult | 0.5B live | 3B mult | 3B live |
|---|---|---|---|---|
| **G-NLL-SMT** (cheap) | **0.84** | **0.81** | **0.83** | **0.84** |
| Baselines | 0.83 | 0.79 | 0.82 | 0.85 |
| Constraint pressure | 0.66 | 0.70 | 0.69 | 0.77 |
| Baselines + CP | 0.81 | 0.80 | 0.75 | 0.82 |

**Verdict:** CP is the **weakest** detector and adding it to baselines **hurts**
(Baselines+CP ≤ Baselines). **H-B1/H-B2 NOT supported on real data.** Log-prob
baselines transfer fine across slices (no collapse); the synthetic "CP wins
cross-family" was a confound artifact. `cp_function` ≈ chance (models are
*confidently* wrong about function choice); function errors are anyway rare
(wrong-valid is ~all arg-value errors). Solid contributions remain: verified
M_t measurement + constraint tax (H-A1) + an honest negative result.

> **Data-integrity note:** the first BFCL run used a required-only grammar that
> **dropped optional args**, making 37% of "wrong-valid" labels artifacts.
> Fixed by adding optional-arg support to `compile_object` (two-state P/Q
> construction, dynamic commas; tests 7/7). All BFCL numbers above are
> regenerated with clean labels — which made the CP-negative verdict *stronger*
> (contamination had inflated CP-only via schema-complexity correlation).

**Distribution-shift check (`cross_bfcl.py`, incl. live_simple: acc 0.33 /
67% wrong-valid — a genuinely harder real slice):** across 6 cross-slice
transfers, log-prob baselines never collapse (0.74–0.83), CP-alone is always
worst, and Base+CP beats the best baseline in only 1/6 transfers by a
within-noise margin (+0.02). The hoped-for "CP is more robust under shift"
advantage does NOT reproduce on real data — the synthetic collapse (baselines
→0.23) was a confound artifact, not a property of shift. Under shift into live,
G-NLL-SMT *alone* degrades most; the robustness comes from the other cheap
log-prob features (max/avg token unc), not CP.

## Multi-sample AST-entropy baseline (D)

`multisample.py` + `add_multisample.py` + `eval_ms.py`: N=8 unconstrained
temperature samples per task, clustered by canonical AST (function + normalized
args), entropy over clusters. Qwen-0.5B:

| Detector | multiple AUROC | live_simple AUROC |
|---|---|---|
| G-NLL-SMT (single) | 0.83 | 0.81 |
| CP (single) | 0.66 | 0.70 |
| **AST-entropy (multi, 8×)** | **0.68** | **0.73** |
| Baselines (logit) | 0.83 | 0.78 |
| Baselines + AST | 0.78 | 0.78 |

**The expensive multi-sample AST-entropy is the *weakest* detector** and adds
nothing to the cheap baselines — replicating the seed paper (Ye et al.): multi-
sample semantic/AST entropy does not beat simple single-sample logit methods in
function calling. Net: on real FC UQ, plain **G-NLL-SMT / logit baselines win**;
neither CP nor AST-entropy improves on them.

## INSIDE / EigenScore internal-state baseline

`inside.py` + `add_inside.py` + `eval_inside.py` adapt the main method from
Chen et al., *INSIDE: LLMs' Internal States Retain the Power of Hallucination
Detection* (ICLR 2024). For each task we draw K=8 free samples, represent every
sample by its last-token hidden state from Qwen's middle layer, form the K x K
covariance matrix, add alpha=0.001 I, and use the mean log10 eigenvalue as the
uncertainty score. AST and INSIDE features are computed from the **same** samples.

Full Qwen-0.5B BFCL multiple result (n=174, 44.8% wrong-valid):

| Raw score | AUROC | AUPRC | Risk@80%cov |
|---|---:|---:|---:|
| G-NLL-SMT | **0.833** | **0.789** | **0.357** |
| AST entropy (same K=8 samples) | 0.695 | 0.654 | 0.371 |
| **INSIDE EigenScore** | **0.641** | **0.560** | **0.414** |
| INSIDE mean-token-pool ablation | 0.634 | 0.566 | 0.400 |

Repeated-CV AUROC: Baselines=0.825, Baselines+AST=0.794,
Baselines+INSIDE=0.795, and Baselines+AST+INSIDE=0.787. Thus EigenScore is a
real but weak signal here: its bootstrap 95% AUROC interval is [0.558, 0.722],
it is only moderately correlated with AST entropy (r=0.415), and essentially
uncorrelated with G-NLL-SMT (r=-0.031), but that distinct signal does **not**
improve the combined detector. It is not a sample-length shortcut
(r=-0.172; sample length itself AUROC=0.565).

This is EigenScore **without feature clipping**, matching the paper's explicit
"EigenScore (w/o)" ablation. The clipping extension modifies activations during
generation and should be evaluated separately rather than approximated by
post-hoc clipping of saved sentence embeddings.

## Agent-trajectory baselines: SAUP, UProp, HTC

`trajectory_baselines.py` adds reusable, equation-level implementations of the
three attached multi-step methods:

- **SAUP-D**: normalized per-response NLL, the plain-distance situational
  weights, and weighted RMS propagation (Eq. 1). The learned SAUP-HMMD variant
  additionally needs domain trajectories/labels to fit its CHMM; the aggregator
  accepts such learned weights when available.
- **UProp**: TDP aggregation, the published Gaussian neighborhood kernel,
  trajectory-dependent PMI (Eq. 8), and step-length normalization (Eq. 9).
- **HTC**: the complete ordered 48-feature map from Appendix D.1, ready for the
  paper's L2 (`HTC-Full`) or L1 (`HTC-Reduced`) logistic calibrator.

`eval_trajectory_baselines.py` evaluates them on a common trajectory JSON
schema; HTC predictions are out-of-fold to avoid train/test leakage. See the
module docstring for the schema and run:

```bash
python eval_trajectory_baselines.py agent_trajectories.json
python -m unittest test_trajectory_baselines.py -v
```

**Applicability warning.** The existing BFCL files contain one generated tool
call per task, not a sequence of agent decisions. Treating JSON token roles as
agent steps would not reproduce any of these papers. On a genuine one-step
trajectory SAUP collapses to the per-call uncertainty, UProp has no extrinsic
uncertainty, and HTC loses its cross-step dynamics. Consequently these methods
are implemented and tested, but a meaningful comparison requires a multi-turn
tool-use benchmark (or a BFCL extension with executed tool observations and
follow-up decisions), plus TDP resampling for UProp.

## Critical high-confidence errors

`critical_errors.py` implements the first actionable item from `plan.md`: count
wrong-valid calls that would still be executed because the detector ranks them
as low-risk. In addition to ordinary selective risk, it reports the absolute
critical-error rate and the fraction of all errors missed by the rejection
policy. It also audits literal model confidence using semantic-token geometric
mean probability (`exp(-avg_sem_tok_unc)`). Legacy result files predate that
field, so the script explicitly falls back to `exp(-avg_tok_unc)` and warns
that JSON syntax tokens can inflate confidence. Newly generated feature files
include `avg_sem_tok_unc` and `n_sem_tokens`.

```bash
python critical_errors.py \
  features_bfcl_multiple.json features_bfcl_live_simple.json \
  features_bfcl_multiple_3b.json features_bfcl_live_simple_3b.json
```

Current result:

| Slice | confidence field | wrong among calls with literal confidence >=.95 | critical errors in safest 20% by G-NLL-SMT | by CP-only |
|---|---|---:|---:|---:|
| 0.5B multiple | semantic-only | 17 / 81 | **1 / 35** | 11 / 35 |
| 0.5B live_simple | semantic-only | 37 / 89 | **11 / 44** | 27 / 44 |
| 3B multiple | legacy all-token | 33 / 147 | **1 / 35** | 11 / 35 |
| 3B live_simple | legacy all-token | 57 / 152 | **9 / 44** | 19 / 44 |

Thus extreme token confidence does not imply semantic correctness, especially
on the live slice. G-NLL-SMT isolates a very safe subset on `multiple`, but not
on `live_simple`; CP is substantially worse at identifying safe calls. The two
new 0.5B runs exactly reproduce all prior IDs, predictions, and labels, changing
only the newly logged semantic-normalized fields. The 3B literal-confidence
column still needs a semantic-only rerun.

## API scale check: OpenRouter native tool calls

`run_bfcl_openrouter.py` is a deliberately separate runner for a larger remote
model. It uses OpenRouter's native `tools` / `tool_choice="required"` interface
and writes an independent, checkpointed artifact; it never overwrites a local
constrained-decoding result. The default is
`qwen/qwen3-next-80b-a3b-instruct`, selected because OpenRouter documents native
tool calling and structured outputs for it.

The current official BFCL files are V4, while the historical local artifacts in
this repository are V3. The runner records `data_version` and defaults to V4;
do not pool the two versions in one result table. The pinned V4 data directory
is passed explicitly through `BFCL_DATA_DIR`.

```bash
# First validate the local task conversion; makes no network/API request.
BFCL_DATA_DIR=./data/bfcl_v4_6ea57973 \
  python run_bfcl_openrouter.py multiple --limit 20 --dry-run

# In a shell that has OPENROUTER_API_KEY configured (never commit the key):
BFCL_DATA_DIR=./data/bfcl_v4_6ea57973 \
  python run_bfcl_openrouter.py multiple --limit 20 \
  --output bfcl_openrouter_multiple_qwen3_next_smoke.json
BFCL_DATA_DIR=./data/bfcl_v4_6ea57973 \
  python run_bfcl_openrouter.py multiple \
  --output bfcl_openrouter_multiple_qwen3_next.json
```

The API cannot expose the local decoder's pre-mask distribution and grammar mask,
so it cannot calculate CP / projection cost. It is a scale comparison for
native structured-call validity, BFCL correctness, wrong-valid errors, latency,
and token usage---not evidence about whether CP improves a logit baseline.

## Scale check (Qwen2.5-3B, bf16)

Combined-detector AUROC (CV), 3B vs the finding holding across scale:

| Slice (clean labels) | G-NLL-SMT | CP-only | Baselines | Baselines+CP |
|---|---|---|---|---|
| 3B multiple | 0.832 | 0.685 | 0.815 | 0.749 (hurts) |
| 3B live_simple | 0.840 | 0.768 | 0.845 | 0.817 (hurts) |

At 3B, CP-only is again clearly below G-NLL-SMT and Baselines+CP **hurts**.
**Scale 0.5B→3B does not change the verdict.** With optional-arg support, 3B is
now more accurate (live acc 0.46 vs 0.5B 0.32), but plenty of wrong-valid
remains and the ranking is unchanged: G-NLL-SMT ≥ Baselines ≫ CP.

## Error-type decomposition & localization (last niches for CP)

`error_types.py`: wrong-valid on BFCL is **almost entirely argument-value
errors** (live_simple: 0 function errors; multiple: 9/174@0.5B, 3@3B). Even on
arg-value errors — CP's supposed home turf — G-NLL ties/beats cp_argvalue
everywhere (e.g. 3B live: gnll 0.886 vs cp_argvalue 0.717). No error-type niche.

`run_localize.py`: in wrong arg-error calls with ≥2 args, which detector's
highest slot = the actually-wrong argument?

| | per-slot CP | per-slot NLL | random |
|---|---|---|---|
| 0.5B multiple (n=34) | 0.71 | **0.91** | 0.49 |
| 3B multiple (n=34) | 0.79 | **0.91** | 0.47 |

Both beat random (localizing the uncertain slot *works*), but **per-token NLL
localizes the wrong arg far better than CP**. So even interpretability/
localization belongs to cheap log-probs. **No niche found where CP wins** except,
in principle, when token log-probs are unavailable (API-only models).

## Limitations / next steps (for the full thesis)

- The general BFCL grammar supports enum/string/integer/number/boolean and
  optional arguments, but not arrays, objects, tuples, or nested schemas.
- Single-step BFCL has been evaluated with Qwen2.5-0.5B and 3B. A first
  stateful BFCL multi-turn smoke is complete with Qwen3-14B (19 trajectories,
  301 executed actions); a larger held-out run is still needed.
- AST entropy and INSIDE have been run on 0.5B, but not replicated at 3B.
- SAUP/UProp/HTC have tested feature/aggregation implementations. HTC has now
  been evaluated on the Qwen3-14B multi-turn artifact (OOF AUROC 0.833), while
  SAUP/UProp still require MC alternatives, situation weights, and the more
  expensive TDP resampling required by UProp.
- The repository still needs a dependency lockfile, experiment manifest, and
  tracked result artifacts before the numbers are independently reproducible.

## BFCL multi-turn stateful rollout

`bfcl_multiturn.py` is the first genuine trajectory runner.  It uses the
official BFCL stateful executor after *every* constrained JSON action, feeds
the returned observation into the next decision, and writes state digests and
the official final trajectory verdict.  The pinned `BFCL_v4_multi_turn_base`
files implement the multi-turn protocol introduced in BFCL V3; current BFCL
distributes that protocol under V4 file names.

```bash
# Structural check only: validates the pinned data and constrained tool mapping.
python bfcl_multiturn.py --dry-run --limit 20

# One cached-model smoke.  --bfcl-root is an official Gorilla checkout whose
# berkeley-function-call-leaderboard directory contains bfcl_eval/.
HF_HUB_OFFLINE=1 python bfcl_multiturn.py --offline --limit 1 \
  --bfcl-root /path/to/gorilla/berkeley-function-call-leaderboard \
  --output features_bfcl_multiturn_base_smoke.json
```

The artifact has action-level `rows` and one-row-per-trajectory
`trajectory_records`.  It immediately supports HTC's token/top-1/top-5
trajectory features.  SAUP and UProp are deliberately withheld until a later
MC-sampling pass supplies their required situation weights and predecessor
alternative-decision distances; substituting greedy values would not reproduce
those baselines.

## Jev pre-execution gate

Jev is evaluated as a bounded decision model, not as another tool-call
generator.  For every recorded BFCL action, `run_jev_bfcl_gate.py` asks the
native Jev `Choice` primitive whether the proposed structured call should be
`execute`d or sent to `review`.  The state contains the user turns, prior
executed actions and observations, available tool schemas, and the proposed
action.  It deliberately excludes the current observation, `wrong_valid`, and
the final trajectory label.

```bash
# Leakage/selection tests and a no-network reconstruction check.
python -m unittest test_jev_gate.py
python run_jev_bfcl_gate.py --limit 5 --dry-run

# Official Jev route through OpenRouter; checkpointed after every request.
uv run --with typesafe-sdk==0.7.2 python run_jev_bfcl_gate.py --limit 5
uv run --with typesafe-sdk==0.7.2 python run_jev_bfcl_gate.py --limit 0

# Equivalent direct TypeSafe route when TYPESAFE_API_KEY is configured.
uv run --with typesafe-sdk==0.7.2 python run_jev_bfcl_gate.py \
  --provider typesafe --limit 5

# Paired comparison on exactly the successfully scored action rows.
python eval_jev_gate.py \
  results/bfcl_multiturn_base_qwen3_14b_smoke.json \
  results/bfcl_multiturn_base_qwen3_14b_jev_gate.json
```

The matching `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY` must be present in the
environment or repository-local `.env`.  The first live request on 2026-10-09
reached the official `/v1/systemone` route but returned HTTP 402 due to
insufficient OpenRouter credits, so no Jev quality result is reported yet.
Brier score and ECE are reported only for Jev's native `P(review)`;
G-NLL-SMT and CP remain ranking baselines and are compared by AUROC/AUPRC and
matched-coverage risk.

### Free open-weight alternatives: Clef and Jeeves

`run_clearml_open_decision_gate.py` runs the same gate from public weights,
without a hosted-model API key:

- `Cloudflare/clef` is the full 27B BF16 Clef model (Apache-2.0), pinned to a
  concrete Hugging Face revision.
- `PostHog/jeeves` is the 9B reasoning decision model, with both weights and
  source pinned.  On the available A100 workers it uses BF16 because Jeeves'
  FP8 kernel requires compute capability 8.9 or newer.

Both jobs use the same deterministic five-action prefix first, so their
probabilities can be compared with Jev, G-NLL-SMT, and CP on identical rows.
They write the same checkpointed result schema consumed by
`eval_jev_gate.py`.

```bash
clearml-task --project "Diploma Thesis Multi-Turn UQ" \
  --name "BFCL gate | Clef-27B | smoke-5" \
  --repo https://github.com/rvz16/diploma_thesis.git --branch main \
  --script prototype/run_clearml_open_decision_gate.py --skip-task-init \
  --requirements prototype/requirements_clef.txt --queue high_q_80 \
  --docker nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 \
  --args backend=clef limit=5

clearml-task --project "Diploma Thesis Multi-Turn UQ" \
  --name "BFCL gate | Jeeves-9B | smoke-5" \
  --repo https://github.com/rvz16/diploma_thesis.git --branch main \
  --script prototype/run_clearml_open_decision_gate.py --skip-task-init \
  --requirements prototype/requirements_jeeves.txt --queue high_q_80 \
  --docker nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 \
  --args backend=jeeves limit=5 max_think=768
```
