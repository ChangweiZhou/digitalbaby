# nano/skip, seventh plan: the coordinator's second rule, a byte that leaves the expectation unchanged does not enter the chain (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at e13990b, with `expect.py` added (the system and kernel of `skip.py` unchanged; the table is numpy only, so there is no compiled part to check). The rule is the coordinating session's, sent to this line during `PLAN_2.md`; it asked for it as this line's third plan, a number already used here. **Status.** Development level, development seed 1; no checker gate, seal or outside review is claimed. Written before any run of `expect.py`.

## 1. The rule, exactly

Beside the relay system, a table of cells that count (after `nano/table/PLAN_1.md`): one cell for each run of the last 1 to 4 bytes that has occurred, counting what follows it. When the byte v arrives after the bytes c, two cells are compared: the cell of the run (c, v), which holds what has followed that run, and the cell of the run c, which holds what followed c (the expectation that was held for v's own place). The comparison is made at the longest run (c, v) of 4, 3 or 2 bytes at which both cells have counted at least 20 times, and is half the sum over the byte values of the differences of their relative counts. Under a threshold theta, v is passed over. What is read: the counts of two cells, the second being the one that was active one byte earlier for the same bytes less the newest. No byte is named; no mismatch; no chance. In the program the counts used within a block of 2,000 lines are those at the block's start.

## 2. My reasoning before the run, which the run is to check

The coordinator's expectation: letters pass the test (after a letter the next byte is again a letter or the number, in nearly the same shares); digits and signs fail it. I agree for the letters (I compute a total variation of 0.01 to 0.05 for them at these lengths, so theta must be above 0.05) and for `+`, `=`, the second digit of each number, the first digit of the second number and the answer's hundreds and units.

**I expect it to fail at the first digit of the first number.** Before it stands the end mark, after which the system expects a digit, any of the ten alike. After it the system again expects a digit, any of the ten alike. The two cells hold the same shares, so the first digit of every line passes the test and is passed over. (The answer's tens digit passes for the same reason, which does less harm.) The digits of a number that cannot be foreseen are, to this test, what the letters are: a byte that is not foreseen and after which the same is expected as before. What sets a digit apart from a letter is that the answer depends on it later, and the expectation of the next byte does not hold that.

## 3. Question, hypothesis, runs

**Question.** Which bytes does the criterion pass over, by kind of byte, and does the system then learn the sums?

**Hypothesis (the coordinator's, as given).** With this clock the system has at least 50% of never-shown pairs at every gap within 1,200,000 lines of `g2`. **Refuted if** under 50% (development seed here; the reported seeds only if it holds on the development seed).

Runs (development seed 1, `g2`): (a) the criterion alone, theta 0.05, 0.1, 0.2, after 10,000, 100,000 and 400,000 lines: the share of bytes passed over for each kind of byte (3,000 fresh lines). (b) The system with this clock, theta 0.1, 1,200,000 lines, with the share passed over by kind of byte at every reading. Beside `clock=none` 52.3 / 1.3 / 0.7, `clock=oracle` 93.7 and the trial clock 95.1 (`PLAN_1.md`, `PLAN_2.md`).

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| V1 | theta 0.1, 400,000 lines: at least 90% of the letters are passed over | 0.75 |
| V2 | theta 0.1, 400,000 lines: at least 90% of the first digits of the first number are passed over | 0.85 |
| V3 | theta 0.1: under 5% of `+`, `=`, the second digits and the second number's first digit are passed over | 0.70 |
| V4 | the hypothesis is refuted on the development seed (under 50% at some gap), the hundreds and tens places being the ones that fail | 0.85 |

**Cost.** About 5 CPU-minutes. **Not run.** The share by place within the gap (the table gives the letters as one kind; added only if the letters are the part that fails); a mended criterion; reported seeds unless the hypothesis holds.

## Outcome of (a), and addition 1 (20 November 2026; the addition written before its runs)

The criterion alone, share of bytes passed over (`res/p7_crit_g2_s1.json`), after 10,000 / 100,000 / 400,000 lines: theta 0.05: letters 0.00 / 0.21 / 0.73, nothing else; theta 0.1: letters 0.11 / 0.94 / 1.00, the answer's hundreds 0.01, nothing else; theta 0.2: letters 0.70 / 1.00 / 1.00, the answer's tens 0.07 / 0.11 / 0.13, nothing else of note.

**V1 held** (1.00). **V3 held** (0.00). **V2 failed: the first digit of the first number is never passed over (0.00), and my reasoning in section 2 was wrong for the texts as this line reads them.** The reason: in these texts (as in `cells2.py` and `hold.py`) nothing is named after the end mark, the first byte of the next line not being a byte the system is asked for. The table therefore has no count of what follows an end mark, the two cells cannot be compared at the first digit, and a byte with no comparison enters. The digit is saved by a convention of the stream, not by the criterion.

*Addition 1.* To separate the two, the same runs with the table counting across the end mark as well (`across=1`; the relay system unchanged, it still does not name that byte): (a') the criterion alone; (b') the system, theta 0.1, 1,200,000 lines, beside (b). Prediction V5 (mine): with `across=1` at least 90% of the first digits are passed over at theta 0.1 after 400,000 lines (chance 0.8); V6: (b') is under 50% at every gap where (b) is over 50% (chance 0.7). V4 I now expect to fail for (b) (the hypothesis to hold on the development seed).

## Addition 2 (20 November 2026, written before its runs): the coordinator's amendment, the next two bytes

Received from the coordinating session after addition 1 was written: its own look (final counts after 40,000 lines, no relay system; "a look, not a claim") found, as addition 1 expects, that with counts across the line end the first digit of a line passes the test of the next byte about half the time, and that a comparison over the next TWO bytes separates: each cell also counts the byte two after its run, and the larger of the two total variations is taken; thresholds 0.2 to 0.3. It asked that this line write its own program from the description and check the shares.

*Runs (development seed 1, `g2`, the table counting across the end mark, at least 20 counts in each of the four cells compared).* (a'') the criterion alone at theta 0.1, 0.2, 0.3 after 10,000, 100,000 and 400,000 lines; (b'') the system, 1,200,000 lines, at theta 0.2 and at 0.3 (both reported, as asked).

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| V7 | (a''), theta 0.2, 400,000 lines: at least 90% of the letters and under 5% of every other kind of byte are passed over | 0.70 |
| V8 | (b'') at theta 0.2 or 0.3: at least 50% of never-shown pairs at every gap, last three readings (the coordinator's hypothesis, development seed) | 0.60 |
| V9 | (b''): the letters are passed over at 90% or more only after 100,000 lines (later than the trial clock, which had them by 100,000) | 0.50 |

## Outcome of (b), addition 1 and addition 2 (entered 20 November 2026; development seed 1; `res/p7_*.json`)

`g2`, 1,200,000 lines, never-shown pairs, mean of the last three readings (the three gaps give one figure once the letters are passed over); the share passed over is over the bytes of the last 100,000 lines unless a reading is named.

| Clock | Never-shown pairs | at 100,000 / 200,000 lines | Letters passed over in the first / second 100,000 lines | First digit of the line passed over | Other kinds |
|---|---|---|---|---|---|
| (b) next byte, theta 0.1, nothing counted after the end mark | **96.6** | 5 / 88 | 0.60 / 0.98 | 0.00 | answer's hundreds 0.008 |
| (b') the same, the table counting across the end mark | **8.1** (by place 73, 18, 99) | 4 / 25 | 0.60 / 0.98 | 0.12, 0.46, 0.81, 0.94 over the first four readings, then 1.00 | answer's hundreds 0.008 |
| (b'') next two bytes, across, theta 0.2 | **95.5** | 57 / 89 | 0.87 / 1.00 | 0.00 | answer's hundreds 0.006 |
| (b'') next two bytes, across, theta 0.3 | **94.2** | 73 / 90 | 0.96 / 1.00 | 0.00 | answer's hundreds 0.021 |
| trial clock, second form (`PLAN_2.md`, B) | 95.1 | 32 / 87 | | | |
| `clock=oracle`, 400,000 lines / `clock=none` | 93.7 / 52.3, 1.3, 0.7 | | | | |

The criterion alone with the next two bytes (`res/p7_crit_g2_s1_across_two.json`), share passed over after 10,000 / 100,000 / 400,000 lines: theta 0.2: letters 0.45 / 1.00 / 1.00, answer's hundreds 0.01, every other kind 0.00; theta 0.3: letters 0.91 / 1.00 / 1.00, answer's hundreds 0.01 to 0.03, every other kind 0.00. With the next byte only and counts across the end mark: the first digit of the line 0.27 / 0.19 / 0.97 at theta 0.1. These agree with the shares of the coordinator's own look.

**The coordinator's hypothesis holds on the development seed** for the rule as amended (next two bytes): 95.5 and 94.2% at every gap, where under 50% would refute. The chain recovers once aligned, and the letters are separated early enough (the system is at 57 and 73% after 100,000 lines, ahead of the trial clock's 32).

**Predictions.** V1 held. V2 failed (see the outcome of (a)). V3 held. V4 failed as first written, for (b), and that is by the stream's convention; the failure I reasoned to is what (b') shows: with the table counting what follows the end mark the first digit is passed over and the system stays at 8.1% (hundreds 73, tens 18, units 99: exactly the places that need that digit). V5 held (0.97). V6 held. V7 held. V8 held. V9 failed (the letters are at 0.87 and 0.96 within the first 100,000 lines).

**Reading (mine).**
1. A rule that reads no mismatch and uses no chance finds the letters on this text, as early as the trial clock or earlier, and costs a table of a few thousand counting cells.
2. What it tests is not whether a byte matters to the answer. It tests whether a byte changes what is expected of the next one or two bytes. On `g2` the two coincide for a reason of the text: a number has two digits, so after its first digit the next two bytes are "digit, sign" and before it they were "digit, digit". The one-byte form shows what happens when they do not coincide: the first digit of the first number looks to it exactly as a letter does (a byte that was not foreseen, after which the same is expected as before), and it is passed over.
3. From this I expect the two-byte form to fail in the same way on numbers of three digits (before the first digit: "digit, digit"; after it: "digit, digit"), and any fixed look-ahead to fail on numbers longer than it. The trial clock reads the answer's mismatch and should not fail there. This is a prediction, not a finding; the next plan tests it.
4. The two rules fail in different places: the expectation rule where a byte that matters later is locally unremarkable; the trial rule where there are many byte values to try (`PLAN_5.md`, running) or where one closing alone earns nothing (`m1`, `PLAN_4.md`, running).

**Cost.** About 8 CPU-minutes. Not run: the reported seeds for this rule (the next plan runs them beside the trial clock on the text that should part the two).
