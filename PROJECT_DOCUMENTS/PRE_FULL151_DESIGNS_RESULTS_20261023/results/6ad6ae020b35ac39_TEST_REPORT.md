# RCE60-4.2 hardened test report

All results in this document are runner/engineering validation unless explicitly described as a native contract check. The fixture is not evidence that a MiniFly mechanism works.

## 1. Reproduced failure mechanism

The old RCE60-4.1 mutator was fuzzed for 10,000 offspring at each requested sigma and checked against V82E's real frozen gene boxes.

- sigma 0.40: 2,450 / 10,000 invalid (24.50%)
- sigma 0.50: 3,449 / 10,000 invalid (34.49%)
- sigma 0.65: 4,235 / 10,000 invalid (42.35%)
- `g_k0` was the most frequent offending coordinate at every sigma.

This directly reproduces the class of error seen on the Mac.

## 2. Hostile unit/invariant suite

`pytest -q`: **12 passed**.

Coverage includes:

- exact 17-gene V82E box contract;
- all known anchors/hybrids donor-locked and valid;
- 34 deliberate one-gene attacks (low/high for every gene) fail closed;
- 22,500 generated offspring = 3 donor families × 3 sigma settings × 2,500 each, all valid;
- historical `sigma * span / 4` units;
- historical per-gene clipping rather than reflection;
- population size/uniqueness checks;
- candidate-ID tampering rejected before worker submission;
- relevance-context activation/deactivation semantics preserved;
- anomaly guard is live;
- hostile preflight genome audit exercised;
- discrete support-toggle provenance is separate from sigma provenance.

The separate semantic regression script also passes and verifies useful-write, revised-useful, collateral-write, and time-only transitions remain separated.

## 3. Exact-final 60-generation engineering campaign

The exact packaged 4.2 source completed:

- hostile preflight: **40 checks passed**;
- preflight genome fuzz: **18,000** generated genomes (3 donors × 3 sigmas × 2,000), plus **34/34** deliberate invalid genomes rejected;
- one complete preflight continuous life;
- all **60 generations**;
- block reduction;
- confirmation;
- final state `COMPLETE`.

Verdict is deliberately `ENGINEERING_TEST_ONLY_NOT_MINIFLY_EVIDENCE` because this execution used the fixture backend.

## 4. Multiprocessing, interruption, and determinism

Using the exact final source on a six-generation fixture campaign:

- serial (1 worker), parallel (2 workers), and generation-3-stop→resume runs all completed;
- final `checkpoint.json` SHA-256 was identical across all three modes: `371f220d9d5d4f06429eb177f9da673015ef6e59b9513a6ef712084855a8f25f`;
- final `FROZEN_CANDIDATES.json` SHA-256 was identical: `70aad0b97f4c27eabf4fa6f326b57bd09fb827b435da86cbab1757780e5a5d52`;
- final `CONFIRMATION.json` SHA-256 was identical: `9d488dec5d98338d1a6350760acb6ce06cee25645568b3c7a5dc7a7004edd1f6`.

## 5. What could not be executed here

A complete native MiniFly life could not be run in this runtime because V82E's inherited `iteration23/iteration24` source dependency tree is not fully mounted alongside the extracted V82E snapshot. The user's actual checkout has that tree (as demonstrated by V86 and the failed 4.1 preflight).

For this reason native 4.2 preflight is deliberately stronger on the user's Mac: it passes every generated fuzz genome through the actual source-locked `model_evo.genome_array` and constructor-smokes fresh offspring before generation 1 is allowed to start.
