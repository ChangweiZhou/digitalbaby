# nano/skip, sixth plan: the letters anywhere before the `=`, inside the numbers as well (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at c5684b1, with the text `g3` added to `skip.py` (`check_skip_output.txt` PASSED on 21 cases, three of them on this text). **Status.** Development level; no checker gate, seal or outside review is claimed. Written before any run on this text except the 300-line cases of the check, and before the outcomes of `PLAN_2.md` C, `PLAN_4.md` and `PLAN_5.md` are known (those runs are in progress or queued).

**Why.** In `g2` all the letters stand in one place, after the `+`, and the second number and the `=` are at fixed delays from the answer whatever the clock does. That is a special help. If the clock is what does the work, the same rule should serve when the letters fall anywhere.

**One change: the text `g3`.** Lines of `aa+bb=ccc` with 0 to 8 letters (four letter values, drawn afresh), each letter put by chance into one of six places: before the first number, between its digits, after it, after the `+`, between the second number's digits, after it. Examples as scored: `1w2y+3w4y=046`, `0zxz7xz+8zx0x=087`. Without a clock no digit of either number is at a fixed delay from the answer unless the places near it happen to be empty. Scoring as before: 500 never-shown pairs with 0, 4 and 8 letters, places and letters fixed by the pair, whole answer right, last three readings.

**System.** As `PLAN_2.md`, second form, nothing retuned. 256 cells, 55,570 numbers that learn, 72 at the entries.

**Runs.** Development seed 1: the rule (1,200,000 lines), `clock=none` (1,200,000), `clock=oracle` (400,000; the check). Then the five reported seeds, 1,200,000 lines: the rule and `clock=none`.

**Hypothesis H6.** With the letters anywhere the trial clock still has at least 50% of never-shown pairs at every number of letters within 1,200,000 lines (median of five seeds). **Refuted if** under 50% at any.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| U1 | H6 holds | 0.70 |
| U2 | `clock=none`: under 5% with 8 letters on every reported seed | 0.90 |
| U3 | on at least four of five seeds exactly the four letters are closed at every reading from 300,000 lines on | 0.60 |
| U4 | no digit is closed at the end of any reported-seed run | 0.75 |

**Cost.** About 22 CPU-minutes. **Not run.** Letters inside the answer; more letter values on this text; `m1` with letters anywhere.

## Outcome (entered 20 November 2026; `res/p6_g3_*.json`; table by `python3 table.py p6_g3 evid,none,oracle <seeds>`)

`g3` (0 to 8 letters anywhere before the `=`), 1,200,000 lines, never-shown pairs with 0 / 4 / 8 letters, mean of the last three readings. Reported seeds in the order 73129, 93683, 116089, 406364, 545353.

| Arm | 0 letters: each seed; median | 4 letters | 8 letters | Seed 1 |
|---|---|---|---|---|
| the trial clock | 96.8, 95.6, 95.9, 95.3, 96.8; **95.9** | the same | the same; **95.9** | 95.7 |
| `clock=none` | 52.8, 52.3, 44.8, 47.7, 50.4; **50.4** | 10.3, 11.7, 10.3, 10.8, 10.3; **10.3** | 1.9, 2.2, 1.9, 1.7, 2.5; **1.9** | 48.3 / 10.9 / 2.1 |
| `clock=oracle`, 400,000 lines | | | | 93.7 |

On all six runs of the rule exactly w x y z are closed at every reading from the first (100,000 lines) to the last; no digit or sign at any reading.

**H6 holds** (95.9% at the median, least seed 95.3, where under 50% would refute). **U1, U2, U3, U4 held.**

Reading (mine). The help that `g2` gave (all letters in one place; the second number at a fixed delay from the answer) is not what the rule lives on. With the letters inside the numbers the letters are found sooner, if anything (by the first reading on every seed, against the first or second on `g2`). I have no tested account of why sooner; one difference is that here a letter can fall between any digit and the answer, so more of each line's answer depends on each closing.

**Cost.** About 20 CPU-minutes.
