# BYTE-LEARNER v8, refined — engineering trial report

**Date:** 25 September 2026.
**Design:** `BYTE_LEARNER_CAUSAL_SHARED_DESIGN_v8_20260925.md`, plus refinements R1–R6 in `REFINEMENTS.md`. That file was written before the run, and its hash is recorded in `RESULTS.json`.
**Status:** engineering only. There are no confidence intervals and no success label, and the 140-life confirmation is still not licensed.

## Verdict

- **The interface works.** Teaching reaches FULL_151 through `CAUSAL_DT`, both clocks stay exact, and reads are causal and read-only.
- **The two implementations agree.** This is a separate implementation from ChatGPT's. On v8's worlds 0 and 1, it reproduces ChatGPT's run exactly.
- **The science is not settled.** TRACE88 writing improves the clean paired probe in all four worlds, but its cross-pair AUC stays weak (0.53–0.57). FE0 is stronger than TRACE88 on every readout in every world.
- **The text head is behind the baselines.** It sits 0.53 bits per byte behind the correctly defined Markov2.

## What ran

- **Worlds:** namespace `BYTE-LEARNER-V8-ENG`. Worlds 0 and 1 are v8's; worlds 2 and 3 are fresh (R5).
- **Trials:** each world runs its first acquisition block: 48 pairs × 6 bouts, giving 576 F3 records of 14 bytes. Bytes are 30/14 s apart and records start 165 s apart. Each world has 24 side-0-aversive pairs and 24 side-0-neutral pairs.
- **Arms:** four, sharing the world and event order: TRACE88 and FE0, each with and without writing (R3 added FE0 without writing).
- **Readouts:**
  - the cross-pair AUC on repeated side-0 events (240 per world);
  - the v8 **sequential** paired choice, whose side-1 read comes after side 0's feedback;
  - the **clean** paired choice fixed in R2: both cues are read from copies of the pre-bout state, with no feedback (240 per world).
- **Runtime:** 77 s in total.

## Checks

All 43 pass:
- **Interface checks (11):**
  - reads are pure;
  - a later outcome cannot change an earlier read;
  - `0xF1` alone never teaches;
  - a mistimed `Teach` is rejected;
  - TRACE88 equals the v6 encoder;
  - the read equals the frozen response.
  - Each has a negative or positive control, and every control fired.
- **World checks (32):**
  - interval accounting: 13 byte commits, 1 teaching event and 1 gap per record; 95,040 s committed; clock error 0;
  - the write and no-write arms get identical codes;
  - writing changes fast and slow memory;
  - the two front ends produce different codes.
- **Cross-implementation (R6):**
  - The 14 pre-specified items from ChatGPT's `V8_TRIAL_RESULTS.json` match exactly: world hashes, AUCs and sequential choices.
  - The 32 post-hoc items from `V8_POSTHOC_DIAGNOSTICS.json` match to within 3.3 × 10⁻¹⁶: FE0 without writing, and the clean probe for every arm.

## Results (descriptive)

Worlds 2 and 3 are the only data on which R2 and R3 were fixed in advance.

| World | Arm | AUC, repeated side 0 | Sequential choice | Clean choice | Clean margin (Hz) | Mean v, aversive / neutral |
|---|---|---:|---:|---:|---:|---|
| 0 | TRACE88 write | 0.566 | 0.608 | 0.654 | +0.336 | +2.34 / +2.74 |
| 0 | TRACE88 no-write | 0.569 | 0.567 | 0.550 | +0.007 | −1.04 / −0.99 |
| 0 | FE0 write | 0.800 | 0.746 | 0.796 | +0.822 | −0.58 / +0.40 |
| 0 | FE0 no-write | 0.513 | 0.417 | 0.412 | −0.062 | −1.38 / −1.36 |
| 1 | TRACE88 write | 0.526 | 0.650 | 0.675 | +0.417 | +2.71 / +2.95 |
| 1 | TRACE88 no-write | 0.530 | 0.438 | 0.446 | −0.041 | −1.14 / −1.06 |
| 1 | FE0 write | 0.637 | 0.742 | 0.775 | +0.848 | +0.43 / +1.06 |
| 1 | FE0 no-write | 0.468 | 0.521 | 0.517 | −0.057 | −1.31 / −1.39 |
| **2** | TRACE88 write | 0.539 | 0.646 | 0.675 | +0.495 | +2.64 / +2.73 |
| **2** | TRACE88 no-write | 0.487 | 0.471 | 0.450 | −0.004 | −1.02 / −1.07 |
| **2** | FE0 write | 0.706 | 0.742 | 0.775 | +0.895 | −0.19 / +0.51 |
| **2** | FE0 no-write | 0.540 | 0.529 | 0.521 | +0.007 | −1.51 / −1.46 |
| **3** | TRACE88 write | 0.550 | 0.704 | 0.704 | +0.527 | +2.13 / +2.45 |
| **3** | TRACE88 no-write | 0.461 | 0.400 | 0.412 | −0.071 | −1.05 / −1.14 |
| **3** | FE0 write | 0.714 | 0.771 | 0.804 | +1.121 | −0.36 / +0.67 |
| **3** | FE0 no-write | 0.421 | 0.404 | 0.421 | −0.057 | −1.39 / −1.53 |

What the table shows, without significance tests (the bouts within a pair are correlated):

1. **Writing helps FE0 everywhere.** In every world, FE0 with writing beats FE0 without writing on all three readouts, and beats TRACE88 with writing.
2. **The TRACE88 effect depends on the readout.**
   - Clean paired choice with writing is 0.654–0.704, against 0.412–0.550 without writing, in all four worlds.
   - The cross-pair AUC difference (write minus no-write) is −0.003 and −0.005 on worlds 0 and 1, but +0.052 and +0.089 on fresh worlds 2 and 3. ChatGPT's report that TRACE88 writing lowers AUC is therefore specific to worlds 0 and 1.
   - Even so, TRACE88's AUC with writing stays between 0.53 and 0.55.
3. **TRACE88 writing shifts every response upward.** Compared with no-write, mean v rises by 3.2–4.0 Hz for aversive and neutral cues alike, while the separation between them stays small (0.08–0.40 Hz). The slow-memory change is about 2–3 times FE0's: L2 243–261 against 80–115. The cause has not been tested. The input scale (L2 against max normalisation) and the overlap of KC codes between cues are candidates, but neither was measured here.

**Text scoring engineering check** (the same two project documents as v8; 10,000 training bytes, 4,000 frozen test bytes):

| Model | Test bits per byte |
|---|---:|
| TRACE88 → KC → 256 head | 7.087 |
| Unigram | 7.106 |
| Order-1 model (labelled "markov2" in the v8 script) | 6.672 |
| **Markov2, last two bytes (R4)** | **6.556** |
| 8-gram, last eight bytes | 6.587 |

The head beats only the unigram, by 0.02 bits per byte. On these 10,000 bytes the correct Markov2 is the best model, ahead of the 8-gram. All scores are finite, each byte was scored before it was seen, and the head stayed frozen during the test. This checks the scoring code only; it is not a language result.

## Limits

- **Sample:** four engineering worlds, a single acquisition block each. There is no 24-hour retention, no revision and no open byte stream.
- **Dose:** the teaching dose is `event(30/14)`, not the old `event(30)`. The v6 probe measured about a tenfold difference in how much plasticity they produce.
- **No-write control:** no-write keeps activity adaptation, as intended.
- **Text:** the two text files are project documents on similar topics.

## Files

`REFINEMENTS.md` (the pre-run specification), `v8_trial.py` (reproduces everything in 77 s: `OPENBLAS_NUM_THREADS=1 <V88 venv python> v8_trial.py`), `RESULTS.json` (28 KB) and this report. ChatGPT's v8 folder was read but not modified.
