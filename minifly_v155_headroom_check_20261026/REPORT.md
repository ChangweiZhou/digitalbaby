V155 HEADROOM CHECK: ROOM_BUT_NOT_COUNTING (development text; pre-design falsifier)

# Headroom check: report (26 October 2026)

**Verdict, by the rules committed before any code ran (`PLAN.md` §3, commit 1315d911):** ROOM_BUT_NOT_COUNTING.
* A learned representation of MiniFly's receptor trace improves on the frozen random keys by 0.26 bits/symbol.
* It does not beat counting at 40 KB.

This is one development text (*Thursday*), one seed per setting, and back-propagation references used only as measuring sticks. Nothing here changes MiniFly.

## Results
*Thursday*, V153's split (40 KB taught, 5 KB calibration, 10 KB held-out); held-out bits/symbol.

| Reading | Held-out | Accuracy | Train (class bits) |
|---|---:|---:|---:|
| T1: best linear readout (R1) of the raw 88-channel receptor trace | 3.463 | 0.312 | 3.39 |
| T2: R1 of the frozen random KC code (V153's cell P81) | 2.963 | 0.426 | 2.00 |
| T3: a hidden layer trained by back-propagation on the same trace | **2.699** | 0.470 | 1.82 |
| T4: the same learner on the explicit last 4 symbols (control) | 2.625 | 0.482 | 2.12 |
| Counting: bigram, PPM-2, PPM-3, PPM-4, PPM-5 | 3.538, 2.986, 2.667, 2.606, 2.616 | | |

Differences, with line-bootstrap SDs:
* T3 − T2 = −0.263 (0.016);
* T1 − T2 = +0.500 (0.018);
* T4 − PPM-4 = +0.018 (0.019).

**Parity and qualification** (all passed):
* T2 reproduces V153's 2.963 exactly, and the ladder matches V153.
* The KC codes equal those of V153's own collector.
* T4 is 2.625 (≤ 2.90 required).
* Neither chosen reference was stopped by the epoch budget.

**Settings chosen on calibration:**
* T1 L2 = 1; T2 L2 = 30.
* T3: width 2048, dropout 0.5, best epoch 36.
* T4: width 512, dropout 0.5, best epoch 58.

All grid rows are in `results/main/SUMMARY.json`.

## What it shows
1. **There is room on the representation side, and output-side changes cannot reach it.** Learned keys reach 2.70 against 2.96 for the frozen random keys (12 SDs). The best output-side fix is bounded by the frozen code's 2.96, so the representation is where the next gain is.
2. **No learner of this kind beats counting at 40 KB, even with explicit context.** Given the last four symbols directly, the back-propagation learner (T4) only ties PPM-4 (+0.02, within 1 SD).
   * So "beat the best counting rung on the same text at 40 KB" is not a reachable yardstick for any learner of this kind. Counting is close to the best that 40 KB supports.
   * T3's training bits (1.82) are far below its held-out bits (2.70): the reference is limited by data, not by capacity.
3. **The random KC layer does useful work on text.** A linear readout of the raw trace is 0.50 worse than of the KC code. Codex V154 found the opposite on arithmetic. On text, the expansion supplies conjunctions of recent symbols that a linear readout of the trace cannot form.
4. **The trace keeps most, but not all, of four symbols of context.** T3 (trace) is 0.07 behind T4 (explicit four symbols). The fade and superposition lose a little; the larger gap is between the frozen code and what the trace allows.

## Limits
* **The reference is dense; MiniFly's keys are sparse.**
  * T3's hidden layer connects every one of its units to all 88 channels.
  * MiniFly's keys connect each of 5,177 KCs to about 5 of 302 projection neurons (1.8% of possible connections), with 184 KCs active at a time.
  * The 0.26-bit headroom is therefore for *dense* learned keys. How much of it is reachable by learning on the native wiring is not measured here. That is the first question for V155.
* **The chosen T3 setting is at the edge of the grid** (the widest width, with dropout). A larger or better-regularised reference might do somewhat better. This check is a lower bound on the headroom, not a ceiling.
* **One development text and one seed per setting.** The T3−T2 difference is large against its line-bootstrap SD, but seed-to-seed variation of the reference was not measured.
* T1 and T2 fits stopped at 300 iterations without formal convergence, as in V153; T2's parity with V153 is exact.

## Cost and records
* 1,559 s of CPU in total (0.43 CPU-h of the 1.0 cap):
  * smoke 54 s;
  * collection 720 s;
  * references 786 s.
* Four attempts, none failed; the log is `runlog/ATTEMPTS.jsonl`.
* No MiniFly learning records. The references are evaluator fits on 40 KB, with up to 60 epochs each.

## Files
* `PLAN.md` (pre-registered), `AUTHORIZATION.json`.
* `collect.py`, `reference.py`, `summarize.py`, `attempt.py`.
* `results/main/LINEAR.json`, `REFERENCE.json`, `SUMMARY.json`, `SUMMARY.md` (arrays are not committed).
* Logs in `runlog/`.
