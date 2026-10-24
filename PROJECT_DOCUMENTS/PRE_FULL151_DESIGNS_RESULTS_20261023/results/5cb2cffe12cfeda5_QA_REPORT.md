# V87 QA report

## Passed locally

- `minifly_V87_core_identification.py` compiles.
- `v86_protocol.py` compiles.
- `RUN_V87.command` passes `bash -n`.
- V84E champion SHA-256 matches `0b4b3b983d67824778900b1c806fae48d500105c7877c86427427d27c123baf6`.
- Bundled V82E archive contains the audited hashes for `model_evo.py`, `common_evo.py`, and `config.json`.
- Variant manifest has 44 unique interventions; no duplicate intervention tuples were found.
- `g_slow_tau` and `g_pnkc_rewire` are verified exactly neutral in V84E and are not included as fake single-gene ablations.
- Only `REF`, `block_REF_representation`, and `gene_REF_g_pnkc_weight` are allowed to escape the V84E B-matrix freeze. All other causal variants are declared frozen-B.
- One-history **all-44-variant** mock campaign completed 44/44 jobs with receipts and aggregation.
- Two-history representative quick mock completed 22/22 jobs with receipts and aggregation.
- Mock aggregation produced the declared life/acquisition/challenge/mechanism contrast tables and final report.

## Real-backend preflight built into the runner

Before a real job is launched, `--prepare-only`:

1. verifies the pinned NumPy/Numba-compatible environment;
2. hash-verifies the audited V82E source;
3. prepares a fresh exact-pattern-excluded PN input bank;
4. instantiates **every selected real variant**;
5. encodes a real PN pair and executes one real learning event for every variant;
6. checks finite diagnostics;
7. verifies that every declared nonstructural intervention has the exact V84E PN→KC fingerprint.

If any check fails, the campaign aborts before the expensive screen.

## Limitation of local QA

The container used to build this package does not contain the user's historical MiniFly checkout (`iteration23/src/common78.py` and `iteration24/src/common79.py`), which the audited V82E source imports. Therefore the real numerical MiniFly backend cannot be executed here. The real-backend instantiation test is intentionally part of the Mac `--prepare-only` step rather than being silently replaced by an approximate local implementation.

## Scientific guardrail

V87 does **not** choose a minimal core automatically. It produces causal/dose/interaction evidence. Joint pruning and physiology re-certification should be run only after the V87 results identify an explicit stripped candidate.
