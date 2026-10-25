# V149 INPUT-CEILING V1: report

25 October 2026. Design: `minifly_v149_input_ceiling_design_20261025/DESIGN.md`. The protocol was locked at commit ed6bcd18 (`LOCK.json`, 62 files hashed) before the main text was scored.

The main run was executed once, on *The Wisdom of Father Brown* (`chesterton-brown.txt`), 19:53–21:00 UTC. The lock still verified after it.

## Question
How much context reaches MiniFly's KC state? The probe reads the native front-end and KC state of an **untaught** FastMemory (V148 settings; no learning) before every symbol. It then asks how well readouts placed outside MiniFly can predict the next symbol from that state:
* **R1**, logistic regression: the ceiling;
* **R2**, an activity-weighted KC vote: a local statistic.

The purpose is to decide which side to fix first. If the state carries no more than bigram-level context, the context side comes first. If it carries more, the output side comes first.

## Main result (held-out bits per symbol; taught 40 KB, calibration 5 KB, held-out 10 KB)
Counting ladder: unigram 4.376, bigram 3.553, PPM-2 3.052, PPM-3 2.802, PPM-4 2.762.

**Readout checks** (the readout can extract context when the code has it). All pass:

| Code | R1 | Target | Check |
|---|---:|---|---|
| last symbol, one-hot | 3.555 | bigram 3.553 | within 0.15 |
| last two symbols, one-hot | 2.982 | PPM-2 3.052 | within 0.20 |
| constant | 4.370 | unigram 4.376 | not below unigram − 0.02 |

**KC states:**

| Arm | State | R1 | R1 accuracy | R2 |
|---|---|---:|---:|---:|
| **PERM-obs** (primary) | observed KC activity after adaptation, V148 front end | **3.767** | 0.281 | 4.234 |
| PERM-x | binary KC code, V148 front end | 3.757 | 0.278 | 4.241 |
| FE0-x | binary KC code, FE0 front end (faster decay, no permutation) | 3.379 | 0.340 | 4.137 |
| PERM-last | binary KC code of the last byte only | 3.554 | 0.283 | 3.757 |

* No arm's chosen L2 lay at an edge of the grid.
* Line-bootstrap SDs of the three contrasts are 0.012–0.016.
* Full tables: `results/main/SUMMARY.md`. Raw data: `results/main/RESULTS.json`.

## Decision and predictions (scored by the locked `summarize.py`)

| | Registered | Measured | Result |
|---|---|---|---|
| Decision | C ≥ B − 0.10 → CONTEXT_FIRST | C = 3.767, B = 3.553 (C is 0.21 worse than bigram, about 14 SDs) | **CONTEXT_FIRST** |
| P1 | C between unigram and B − 0.10 | 4.376 > 3.767 > 3.453 | **As predicted** |
| P2 | PERM-last within 0.15 of bigram; PERM-x not better than PERM-last by more than 0.10 | PERM-last 3.554 (bigram 3.553); PERM-x 0.20 *worse* than PERM-last | **As predicted** |
| P3 | R2 at least 0.30 worse than R1 on PERM-obs | 0.47 worse | **As predicted** |

The same result held on all three dev texts (Buster, Bryant, Edgeworth; `cycles/CYCLE2.md`, `CYCLE3.md`).

## What this shows
1. **MiniFly's state carries less than bigram-level context.** Even a best-possible readout of the V148 front end's KC state predicts the next byte 0.21 bits worse than counting byte pairs. Fixing the output side alone could therefore not reach the bigram rung with this front end. The context side has to change first.
2. **The long trace hurts.** The full FE-PERM trace is 0.20 bits *worse* than a code built from the last byte alone. The slowly decaying (0.95 per byte), permuted sum of many bytes blurs the most recent byte, and no usable history makes up for that. Adaptation changes almost nothing (3.767 against 3.757).
3. **A faster-decaying front end already does better.** FE0, the other native front end, reaches 3.379: 0.17 better than bigram, though still 0.33 worse than PPM-2. On the dev texts this gap was 0.33, 0.06 and 0.20. It was recorded before the run as a dev-informed expectation, not a prediction, and it held on the main text. A recency-weighted mix of bytes carries usable context beyond the last byte, so the context side is not hopeless.
4. **A simple local readout loses most of it.** The KC vote (R2) is 0.20–0.76 bits worse than R1 on every arm (0.20 on the last-byte code, 0.47–0.76 on the trace codes, where it stays close to unigram). Whatever replaces the output side has to combine KCs, not average them.

## What this does not show
* **R1 is an evaluator-side ceiling, not a MiniFly mechanism.** It shows what the state contains, not what MiniFly can learn to read.
* **This is a ceiling for one readout family**, logistic regression on the state, with 40 KB taught. A richer evaluator readout could extract more.
* **Only the untaught state was measured.** In V148, teaching also raised homeostatic thresholds, which can change the codes.
* **There is one main text**, with no seed variation: everything is deterministic.

## Bearing on the next step (for discussion, no recommendation in this report)
* The decision is CONTEXT_FIRST.
* FE0's result suggests the decay and mixing of the front-end trace are a direct lever.
* The context-side mechanisms reviewed earlier can now be judged against this ceiling, using the same probe, before any native learning:
  * the temporal context model (human SYS-10);
  * Oja plasticity on the input-to-KC connections (fly §2);
  * biased KC sampling (fly L5/L7);
  * a KC clock (fly L8).

## Cost and accounting
* **Main run:** 4,052 CPU-s (states 269 s). Peak memory 1.4 GB.
* **Cycles:**
  * cycle-1 tiny run 92 s;
  * dev run 4,629 s;
  * convergence check about 1,900 s;
  * rehearsals 3,398 s and 4,796 s.
* **Total:** about 5.2 CPU-hours.
* **Failed attempts:** none crashed. Rehearsal 1 found an L2 grid-edge defect; the grid was widened and a second rehearsal was run on a fresh text before the lock (`cycles/CYCLE3.md`).
* **Design corrections made before any run:** the main text's title (*The Wisdom*, not *The Innocence of Father Brown*); the raw-trace arm was dropped, since it gives codes identical to the current trace when thresholds are zero (checked); an empty-state indicator was added.

## Files
* **Code:**
  * `corpus.py`, `states.py`, `readouts.py`, `run.py`;
  * `checks.py` (18/18);
  * `summarize.py`, the locked scorer;
  * `make_lock.py`, which writes `LOCK.json`.
* **Results:** `results/main/`, `results/c3_rehearsal2/`, `results/c3_rehearsal/`, `results/c2_dev/` and `results/c1_tiny/`, each with its log.
* **Cycle records:** `cycles/CYCLE1.md`, `CYCLE2.md`, `CYCLE3.md`, `SETTINGS.json` and `converge_check.*`.
