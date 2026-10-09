# Jev pre-execution gate: implementation status

Date: 2026-10-09

## Research question

Can Jev's native bounded decision distribution identify a schema-valid but
semantically wrong BFCL action before that action mutates the agent state?

Jev is used as an `execute` versus `review` gate over the action proposed by
Qwen3-14B.  It is not used to generate arbitrary arguments.  This keeps the
comparison aligned with Jev's decision-only interface and with the thesis'
selective-execution objective.

## Protocol

- Population: 301 executable actions from 19 BFCL multi-turn trajectories.
- Input available to Jev: user turns through the current turn, prior executed
  actions and their observations, available tool documentation, and the
  proposed action.
- Held out from Jev: the current action's observation, `wrong_valid`, gold
  calls, and final trajectory success.
- Primary target: action-level `wrong_valid`.
- Paired baselines: G-NLL-SMT, CP max, and CP argument-value score on the exact
  same successfully scored rows.
- Metrics: AUROC, AUPRC, selective risk at 20/50/80% coverage, native 0.5-gate
  coverage/risk, Jev Brier/ECE, latency, and token usage when returned.

The runner uses a deterministic shuffled prefix (`seed=2026`) for smoke tests,
checkpoints after each API response, and can resume the same selection.
Successful actions are skipped on resume; provider failures remain auditable
but are retried.

## Verification completed

- Four unit tests pass, including prefix stability and absence of current
  observation/label leakage.
- A five-action dry run reconstructs BFCL state and tool documentation.
- The first live request reached OpenRouter's official TypeSafe-compatible
  `/v1/systemone` endpoint with `~typesafe/jev-latest`.

## Current blocker

The live request returned HTTP 402: the configured OpenRouter account has
insufficient credits.  Therefore there are currently zero successful Jev
predictions and no quality comparison to report.  Resume only after adding
OpenRouter credit or configuring an official TypeSafe API key/endpoint; do not
interpret the provider failure as an experimental result.

## Open-weight alternatives

The provider blocker does not apply to two Jev-compatible open models:

- `Cloudflare/clef` is a 27B decision model under Apache-2.0.  The experiment
  uses the full BF16 checkpoint, not Clef-flash.
- `PostHog/jeeves` is a 9B reasoning decision model with a Jev-compatible
  response.  The experiment uses BF16 because the available A100 GPUs do not
  meet the model's compute-capability requirement for its FP8 CUDA kernel.

Both local runners use the identical selected actions, state builder, gate
wording, labels, and evaluation code.  Initial runs are five-action integration
smokes; they are not sufficient for a quality claim.

The smoke tasks started in parallel on 2026-10-09:

- Clef: `142364cfb0474bd2bcf192743545db6c` on `aiagent01:gpu0`.
- Jeeves: `ba11d0485e8d4501983aad9a966249e9` on `aiagent02:gpu0`.
