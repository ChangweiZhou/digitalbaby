# BYTE-LEARNER v8 — refinements fixed before this run

**Date:** 25 September 2026. Written before any code in this folder ran; its SHA-256 is recorded in `RESULTS.json`.
**Base design:** `BYTE_CORE_NEXT_STAGES_20260922/BYTE_LEARNER_CAUSAL_SHARED_DESIGN_v8_20260925.md`.
**Status:** engineering trial only. There are no confidence intervals and no success labels, and nothing here licenses the 140-life confirmation.

ChatGPT ran an earlier implementation of v8 (`BYTE_LEARNER_TRIAL_V7_20260925/`, `run_v8.py` and `trial_integrated.py`). This is a separate implementation. It does not overwrite those files and does not read their results, except to cross-check (R6).

## Kept exactly as in v8

- **Worlds:** namespace `BYTE-LEARNER-V8-ENG`, block `TRIAL`, indices 0 and 1, R2, massed.
- **Trials:** the first acquisition block only: 48 pairs × 6 bouts, in generator order.
- **Record format:** 14-byte F3 records. Bytes are spaced Δ = 30/14 s apart, and records start 165 s apart. The wrapper is the cue's presentation count mod 2, as in the frozen `run_life`.
- **`CAUSAL_DT` rule:**
  - each byte's interval is committed once;
  - an interval ending in an outcome byte is committed as `event(Δ, pre-outcome KC, r, plastic)` when `Teach(r, t)` arrives at that byte's time;
  - every other interval is committed as `rest`;
  - both clocks align at the end of each record.
- **Readout:** raw reader-relative response `v`; aversive = lower `v`.
- **Text trial:**
  - sources: the same two Markdown files as v7/v8, with hashes checked against the v8 record;
  - data: the first 10,000 bytes of one file for training and the first 4,000 of the other as the frozen test;
  - head: the v6 `TRACE88→KC→256` head with its preflight learning rates;
  - baselines: α = 10.

## Refinements

| ID | Change | Reason |
|---|---|---|
| R1 | The pre-outcome read is computed entirely on copies of the front end and the brain. This includes committing any pending interval and passively advancing to the read time. The real learner is never touched; its state digest is checked before and after | v8 requires a read-only read. The v8 script committed the pending interval on the real brain and advanced the real front end before reading. Both are harmless in order, but they are not read-only |
| R2 | **Clean two-sided probe, fixed now as a readout.** At the start of each bout, before side 0 is shown, both cues of the pair are presented to separate copies of the same pre-bout state. Each copy receives the 12 context bytes (same wrapper rule, start time t0) and is read at t0 + 12Δ with no outcome and no teaching. Margin = v(neutral cue) − v(aversive cue); choice correct if margin > 0, 0.5 if tied. Bouts 2–6 of each pair: 240 per world | In the v8 paired readout, side 1 is read after side 0's outcome and teaching, so it mixes in the feedback that just arrived. That readout is still reported, labelled **sequential** |
| R3 | **Fourth arm: FE0 without writing** (`plastic=False`, same input and clock) | Without it, FE0's write effect has no same-input control. The 2 × 2 design is TRACE88 / FE0 × write / no-write |
| R4 | **Text baselines as v6 defines them.** Markov2 uses the last **two** bytes and the 8-gram the last eight. A context order is used only once the stream holds that many bytes, in both training and test. The order-1 model is also reported, as `markov1` | The v8 script's "markov2" used one byte of context. It also looked up contexts shorter than the model order at stream start |
| R5 | **Fresh replicate worlds 2 and 3** (same namespace, block and code), run after worlds 0 and 1 | They are the first worlds where R2 and R3 were fixed before the data existed. On worlds 0 and 1, both were computed after the fact in ChatGPT's diagnostics |
| R6 | **Cross-implementation check.** On worlds 0 and 1, compare this run's AUC and sequential paired accuracy for the three original arms with ChatGPT's `V8_TRIAL_RESULTS.json`, and its FE0-no-write and clean-probe values with `V8_POSTHOC_DIAGNOSTICS.json`. Any mismatch is investigated before reporting | Two independent implementations of the same protocol should agree exactly. A disagreement means one of them is wrong |

## Not changed (deliberately)

- **The teaching dose `event(30/14)`.** It is not the old `event(30)`, and it is not recalibrated here.
- **The readout family.** No new readout is fitted.
- **The text sources, α and the head's learning rates.**
- **The no-write semantics.** No-write still commits the same events with `plastic=False`, so activity adaptation continues.

## Checks run with every trial

Each check has a negative control or a positive control:
- **Read purity.** NC: an impure read, which commits on the real learner, must change the digest.
- **Reads do not see the future.** Changing a record's outcome byte does not change its earlier read. NC: a read taken after the outcome byte and teaching does differ.
- **An `0xF1` byte alone never teaches.** PC: `Teach(1)` changes fast and slow memory.
- **A mistimed `Teach` is rejected.**
- **Interval accounting.**
  - Per record: 13 byte-interval commits, 1 teaching event and 1 gap rest.
  - The committed time totals 165 s.
  - The brain and sensory clocks agree with the world clock.
- **Same codes with and without writing.** For each front end, the write and no-write arms have identical pre-outcome codes. PC: TRACE88 and FE0 codes do differ.
- **Writing changes the learner.** The write arm's fast and slow state differs from no-write. NC: two no-write runs are identical.
- **TRACE88 is the v6 encoder.** This front end's codes equal those of `preflight.Trace88`. NC: a different seed gives different codes.
- **Text scoring.** The head scores each byte before seeing it, and its parameters are unchanged during the test. The baselines are trained on exactly the same bytes as the head.
