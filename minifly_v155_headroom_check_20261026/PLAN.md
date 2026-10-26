V155 HEADROOM CHECK: PLAN (written and committed before any code was run)

# Headroom check before the V155 design (26 October 2026)

**Status:** a pre-design falsifier, development text only. It measures; it does not change MiniFly. A design document for V155 follows it, whatever the outcome.

## 1. Question
V153 (development, *Thursday*, 40 KB taught) found that the frozen, untaught KC code of the best front end (scrambling on, fade 0.807 per byte) reaches **2.963** bits/symbol with the best linear readout (R1). Counting reaches PPM-2 2.986, PPM-3 2.667, PPM-4 2.606.

No output-side change can beat counting from that frozen code: every readout MiniFly could use is linear in the KC code, and R1 is the best of that family found. Only a changed representation (the keys from the receptor trace to the KCs) can. The adopted step 2 is to learn the keys with a local rule.

Before building any local rule, this check asks: **at 40 KB, could any learned representation of the same 88-channel receptor trace beat counting?** If a hidden layer trained by back-propagation cannot, a local rule will not either, and V155 should not be a key-learning round.

## 2. Readings
All on *Thursday* with V153's split (40 KB taught, records 0–705; 5 KB calibration, 705–790; 10 KB held-out, 5296–5474), V153's class definition (symbols seen in the taught text plus OTHER) and V153's bits function. Settings are chosen on calibration only. Held-out is scored once for each chosen setting.

| | Input | Learner | Role |
|---|---|---|---|
| T1 | the 88-channel receptor trace that the keys read (`FEPerm.read()`, P81 cell, at the native read time), plus an empty-state indicator | R1 (V149 `fit_r1`, L2 grid 10, 1, 0.1; 300 iterations) | interpretation only: does the random KC layer add anything on text? (Codex V154: on arithmetic, with scrambling, it did not) |
| T2 | the frozen random KC code of the same cell (5,177 units) | R1, V153's exact grid 100, 30, 10; 300 iterations | parity with V153's 2.963 |
| T3 | the same 88-channel trace as T1 | one hidden layer trained by back-propagation, in place of the random keys | **the decision reading**: what a learned representation of this trace could reach at 40 KB |
| T4 | the last 4 symbols, written out explicitly (one-hot per position, line start padded) | the same learner and grid as T3 | positive control: can this learner use explicit context at 40 KB at all? |

**T3/T4 learner.**
* Input → hidden layer of width H with ReLU → softmax over the classes.
* AdamW, learning rate 1e-3, weight decay 1e-4, batch 256, seed 0, one CPU thread.
* Up to 60 epochs, stopping after 6 epochs without improvement in calibration bits. The epoch with the best calibration bits is kept.
* Grid: H ∈ {512, 2048} × dropout on the hidden layer ∈ {0, 0.5}. The setting with the best calibration bits is chosen; every setting's calibration score is reported.
* H is not matched to the 5,177 KCs. That is for cost; it is a reference, not a replica.

Back-propagation is used only to measure. It never feeds MiniFly, and no part of this check becomes a candidate learner.

## 3. Rules fixed before running
**Parity (else stop and report):** T2 reproduces V153's 2.963 within 0.005. The PPM ladder reproduces V153's values within 0.001.

**Reference qualification (else verdict REFERENCE_NOT_READY):**
* T4 held-out ≤ 2.90: the learner extracts clearly more than the frozen code from explicit context.
* For the chosen T3 and T4 settings, the best calibration epoch is before epoch 60 (not cut off by the budget).

**Decision, on T3 held-out** (thresholds from V153's recorded numbers):

| T3 held-out | Verdict | Meaning for V155 |
|---|---|---|
| ≤ 2.556 (PPM-4 − 0.05) | LEARNED_CAN_BEAT_COUNTING | a learned representation of this trace can beat counting at 40 KB; V155 may compare local key-learning rules against the frozen keys |
| 2.556 < T3 ≤ 2.863 (frozen − 0.10) | ROOM_BUT_NOT_COUNTING | learned keys can improve on the frozen code but not beat counting at 40 KB; the owner decides between more text and a carry-over / sample-efficiency yardstick before V155 is designed around keys |
| > 2.863 | NO_ROOM_AT_40KB | no key-learning round at 40 KB; the limit is the trace or the amount of text |

Reported alongside, with line-bootstrap SDs (1,000 resamples of held-out lines): T3 − T2, T1 − T2 and T4 − PPM-4.

## 4. What this cannot show
* A negative result holds for this reference learner at 40 KB, not for all learners. Deeper or better-tuned learners might do better.
* A positive result is an upper reference. Local rules usually reach less than back-propagation on the same architecture.
* One development text, one seed for each setting. *Thursday* was already used for V153's rehearsal; *Persuasion* stays untouched.
* Bits on the same text are not carry-over. Carry-over is a V155 design question.

## 5. Cost and limits
* Estimate: about 0.6–0.8 CPU-h. That covers states for one cell (about 2 minutes), T2 (about 7 minutes), T1 and the ladder (a few minutes) and T3/T4 (about 8 MLP fits, at most about 35 minutes on one thread).
* Cap: 1.0 CPU-h, as stated to the owner. If the measured cost projects past the cap, the run is stopped and the owner asked.
* One tiny smoke run (2 KB) checks the code end to end first; its cost counts.

## 6. Execution
* Two processes, because the torch environment cannot load MiniFly (no numba):
  * `collect.py` (MiniFly environment `/tmp/venv_n`) writes the states, T1, T2 and the ladder;
  * `reference.py` (torch environment `/tmp/venv_t`) reads the saved arrays and fits T3/T4.
* `summarize.py` applies section 3 and writes `REPORT.md`.
* Every attempt is logged in `runlog/ATTEMPTS.jsonl` with its measured CPU.
