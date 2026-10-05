# Pinned BFCL multi-turn base slice

Source: `ShishirPatil/gorilla`, commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`,
directory `berkeley-function-call-leaderboard/bfcl_eval/data`.

The BFCL multi-turn protocol originated in BFCL V3.  In the current official
repository its files are named `BFCL_v4_multi_turn_*`; this pinned copy uses
`BFCL_v4_multi_turn_base.json`, its matching ground truth, and the function
documents needed to construct constrained schemas.

SHA-256:

- data: `1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a`
- answers: `1fee67823b317571649177dd89d63969feaae4e810cc7448ee55ba797fb7c8fc`

`bfcl_multiturn.py` deliberately calls the official BFCL executor for state
transitions and final scoring.  Install `bfcl_eval` or pass a checkout through
`--bfcl-root`; the JSON data alone is not an executable environment.
