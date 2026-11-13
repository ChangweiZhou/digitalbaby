# Cycle 1: implement, audit, test run, exercise (24 Oct 2026)

## Refine (implementation)
* `lb1.py`: opponent-coded banks of canonical native compartments (reusing the v0 `fastmem.Compartment` and FE-PERM front end unchanged), shared bank with v0 one-sided homeostasis, learned layer (H=128, W2 and feedback G fixed random), three-factor update, warmup schedules, read-only probes, code extraction for meters.
* `data.py`: orbit splits (15 held / 40 train orbits; every digit in at least 4 train orbits), tasks MOD and COPY, fixed derangement for SHUF, interleaved per-pass shuffled schedule.
* `run.py`: histories with checkpoints (accuracy of system, shared bank and learned bank on all 100 prompts; kernel meters; code drift; CPU).
* `summarize.py`: per-history table and the decision rule.
* Environment: Python 3.11 venv with numpy 2.2.6, scipy 1.14.1, numba 0.61.2, pandas (native dependencies of minifly_a_v3).

## Audit
* Unit checks (`checks.py`, results `CYCLE1_CHECKS.json`): 34/34 pass. Split disjoint and complete (D1: 74 taught, 26 held ordered prompts); schedule never contains held prompts; derangement has no fixed point; all compartments identical at birth; opponent margins exactly zero at birth for every arm; probes leave model state hash unchanged; probe codes equal offline `trace_for` codes; 100 distinct prompt codes in every bank (ORACLE's learned bank is class-coded by construction and is excluded from that gate); repeated prompt gives identical code; two teaching records move every bit margin toward its target in both banks; W1 changes only in LEARN; zero teaching coefficient is exactly a passive native event.
* Read-through findings:
  1. The design's decision rule named "held-out kernel-target alignment". With 26 held prompts and 10 answer classes a held-only alignment is unstable. The implemented held-out kernel meter is the **held transfer gap**: for each held prompt, mean Jaccard to taught prompts with the same answer minus mean Jaccard to taught prompts with a different answer. This is the similarity the readout needs in order to generalise. The all-prompt centred alignment is still recorded. To be written into the lock.
  2. At birth the learned-bank alignment is already positive (0.157) because mirrored prompts (a+b, b+a) share bytes and answers; the held transfer gap is negative (-0.023) because held prompts resemble taught prompts that share a digit but have a different sum. Generalisation requires the gap to become positive.
  3. Address drift: even with the learning rate still ramping (about 1/7 of eta0 by record 148), learned-bank codes of taught prompts moved to Jaccard 0.95 then 0.92 between passes. At full rate the store may not keep up. Cycle 2 must measure drift against taught accuracy.
  4. Weight decay lam=1e-3 at eta0=0.003 is negligible (about 2% over 6,000 records); cycle 2 should decide whether it is meant to matter.

## Test run (development world D1, T-MOD, LEARN and FROZEN, 2 passes, 148 records each)
| Arm | Held (learned bank) | Taught (system) | Gap L birth to end | Min drift |
|---|---:|---:|---:|---:|
| FROZEN | 0.077 | 0.162 | -0.023 to -0.023 | 1.000 |
| LEARN | 0.115 | 0.162 | -0.023 to -0.024 | 0.924 |

Too short to say anything about learning; it shows the pipeline works and gives costs.

## Exercise (end to end)
`run.py` to JSON, then `summarize.py` to table and verdict: works. Cost: 0.12 CPU-s per training record (32 native compartments), 6.5 CPU-s per 100-prompt probe, 240 MB RSS per process. Estimated main history (80 passes, about 5,900 records, 9 probes): about 13 CPU-minutes.

## Revisions carried into cycle 2
* Use the held transfer gap as the held-out kernel meter (write into the lock).
* Measure drift and taught accuracy at full learning rate; tune eta0, lam and passes on development worlds only.
