# API-Bank level-1 data snapshot

Source: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank

API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs, Minghao Li et al.,
EMNLP 2023. https://arxiv.org/abs/2304.08244

`apibank_level1_faf75467.json` is a mechanically reformatted snapshot of the
official `lv1-lv2-samples/level-1-given-desc` dialogues and literal API metadata.
It contains hashes of each original source file. No dialogue text or API
parameter declarations were rewritten. License: Apache-2.0; see
`APIBANK_LICENSE.txt` copied from the official `api-bank/LICENSE`.

Our experiment filters entire dialogues whose tool inventory cannot be
represented by the current scalar grammar. It also reports and excludes
reference calls that violate those declarations. The resulting population is
290 calls from 163 dialogues and 39 target APIs, out of 389 raw API calls.

This is teacher-forced next-call prediction with strict typed API name and
argument matching. It is not the execution-based API-Bank leaderboard score or
an autonomous trajectory rollout. The supplied API inventory follows the
official level-1 given-description protocol (all APIs used in the dialogue).
