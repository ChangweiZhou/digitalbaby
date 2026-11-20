# nano/skip, second plan: each byte value tries being left out of the chain and keeps what lowers the mismatch (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at 667ea43 (the first plan and its outcome), with `clock=trial` added to `skip.py` and one output added to `kern_s.c` (the sum of the answering cell's squared mismatches); `check_skip_output.txt` PASSED on 15 cases, two of them with this clock. **Status.** Development level; no checker gate, seal or outside review is claimed. Written before any run with this clock except the 300-line cases of the check.

## 1. What this follows

`PLAN_1.md`: with the letters passed over by the program the system has 93.7% of never-shown pairs at every gap; with no clock 52.3 / 1.3 / 0.7% at gaps of 0 / 4 / 8 letters after 1,200,000 lines. The sum of the sizes of the strengths held from a byte value's chain lines does not separate the letters, because the cells listen to the letters for where the line is. What differs between a letter and a digit is what leaving it out does to the mismatch.

## 2. The system and the one change

The system of `PLAN_1.md`, `lam1=0`: 256 cells in between, 180 lines, 55,570 numbers that learn. **The change: `clock=trial`, 18 more numbers that learn (one for each byte value).**

*What the rule reads and where it lives.* Each byte value v has an entry to the chain. The entry holds one number u(v), its tendency to be closed, and its present state, open or closed. Every P = 256 bytes of the stream every entry draws its state anew: closed with chance c(v) = logistic(u(v)), kept between 0.01 and 0.99. While an entry is closed its byte value is passed over (it is still on its delay-0 line while it is the present byte). The answering cell sends out one number at every byte, the same to every entry: the sum over its branches of its squared mismatch. An entry averages that number over the period (r), keeps a running mean of r (b) and of |r - b| (w), both with rate 0.02 a period, and at the period's end moves its tendency by

  alpha * z * (1 if it was closed, else 0, less c(v)),  z = (b - r) / w kept within -3 and 3.

So an entry that was closed during a period with less mismatch than usual becomes more inclined to close, and the reverse. Every entry starts at c = 0.2. The rule names no byte and no kind of byte, the program does not tell the system which bytes are letters, and no strength is read. What is not local to an entry is the one number sent out by the answering cell, which is the same for all entries (a signal that reaches many cells; Mr. Xu's sentence quoted in `PLAN_1.md` lists "ability to output signals (chemicals)" among the kinds of cells that may be tried).

*Numbers chosen by hand:* P = 256, the starting chance 0.2, the floor 0.01, the rate 0.02, and alpha. alpha is chosen on the development seed, on `m1` first; if `g2` needs another value it is reported as tuned for the text.

*Scoring.* As in `PLAN_1.md`, with every entry in its likelier state (closed where u > 0) and no drawing. The stream itself is read with the drawing throughout.

*Reading (mine), before the runs.* Nothing rewards closing a letter until the cells answer some lines that are aligned, and at first only the lines with no letters are. Closing one letter aligns the lines whose letters are all that letter (about one line in 27), so the first evidence is weak, and it grows with every letter closed. Closing a digit removes what the answer depends on and should be punished early. `+`, `=` and the end mark may be neutral and wander.

## 3. Question, hypothesis, what would refute it

**Question.** Does this rule find the letters, and soon enough?

**Hypothesis H2 (the line's, as given).** On `g2` the system with the trial clock has at least 50% of never-shown pairs right at every gap within 1,200,000 lines (median of the five reported seeds, mean of the last three readings). **Refuted if** the median is under 50% at any gap.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| Q1 | `m1`, development seed, 200,000 lines: for some alpha of 0.02, 0.05, 0.1 at least 90% of the sums shown at every gap | 0.50 |
| Q2 | `m1`, at that alpha: at the end the four letters are closed and no digit is | 0.50 |
| Q3 | `g2`, development seed, 1,200,000 lines: the four letters are closed at the end | 0.45 |
| Q4 | H2 holds | 0.30 |
| Q5 | the closing by chance with tendencies that never change (alpha = 0): under 20% at the gap of 8 letters on every reported seed | 0.85 |
| Q6 | four byte values closed by chance from the start: under 20% at the gap of 8 letters on at least four of five seeds | 0.90 |
| Q7 | no digit is closed at the end of any reported-seed run of the rule | 0.70 |

## 4. Runs

A. `m1`, development seed 1, 200,000 lines: alpha 0.02, 0.05, 0.1, and alpha = 0. (Beside `clock=none` 20.4 / 22.9 / 10.0 and `clock=oracle` 100 of the first plan.)
B. `g2`, development seed 1, 1,200,000 lines: the alpha chosen in A; if it fails, at most two further settings, said to be tuned on this text.
C. `g2`, the five reported seeds (73129, 93683, 116089, 406364, 545353), 1,200,000 lines, four arms at equal size and length: the rule; `clock=none`; alpha = 0; `clock=values:n` with n the number of byte values the rule most often ends with closed. Where the rule and a control are within ten points, the seeds 1001 to 1010 are added and all fifteen reported.

**Cost.** About 2 CPU-minutes for a run of 1,200,000 lines; about 50 CPU-minutes in all. **Not run.** Other periods, floors and starting chances than those above unless B fails; more cells; the first plan's quantity again; receiving cells that withdraw.

## Outcome of A (entered 20 November 2026; development seed 1; `res/p2_m1_trial_a*.json`)

`m1`, 200,000 lines, sums shown at gaps 0 / 4 / 8, last three readings: alpha 0.02: 27.1 / 27.9 / 11.2, nothing closed; alpha 0.05: 37.5 / 40.0 / 22.5, the letter z closed from 40,000 lines; alpha 0.1: 37.5 / 39.6 / 23.3, z closed from 40,000 lines; alpha 0 (closing by chance): 16.7 / 17.5 / 13.3. Beside `clock=none` 20.4 / 22.9 / 10.0 and `clock=oracle` 100.

**Q1 failed** (no alpha reaches 90%). **Q2 failed** (one letter closed, not four; no digit closed). What the tendencies did (Reading, mine): every digit, `+`, `=` and the end mark moved firmly to open (u from -1.4 to between -3 and -9), so closing them is punished early and clearly, as expected. One letter closed and the scores then began to rise (50 / 48 / 32 at the last reading with alpha 0.05, still rising). The other three letters moved to open as well, and so did `-`, a byte value that never occurs and for which the mismatch says nothing: -1.6, -3.4, -4.4 at the three alphas. So a tendency that receives no consistent evidence does not stay where it is; it wanders by the noise in the period's mismatch, and once near the floor it is tried only one period in a hundred and stays there. The letters' evidence is weak until some lines are aligned, and by then three of them had wandered to the floor.

## Addition 1 (20 November 2026, written before its runs): the same trial, the evidence kept as two means

*One change, to the form of what an entry keeps; what it reads is unchanged* (its own state in each period and the one number sent out by the answering cell). An entry keeps four numbers: the sum and the count of z over the periods in which it was closed, and over those in which it was open, all four fading by 1/3000 a period. Its tendency is u = logit(p0) + alpha * t, where t is the difference of the two means of z (closed less open) divided by the root of (1/(count closed + 1) + 1/(count open + 1)): the difference in units of its own uncertainty. An entry with little evidence therefore stays near p0 and keeps trying, and one that has not been closed for long loses its evidence and tries again. 72 numbers at the entries where there were 18.

*Runs (development seed 1).* `m1`, 200,000 lines, `form=evid`: (alpha, p0) = (2, 0.2), (2, 0.5), (4, 0.2), (4, 0.5). The best of these goes to B and C in place of the first form; the first form at alpha 0.05 is run in B beside it.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| Q8 | `m1`: with some of the four settings the four letters are closed at the end and no digit is | 0.55 |
| Q9 | `m1`: with that setting at least 90% of the sums shown at every gap | 0.40 |

## Outcome of addition 1 (entered 20 November 2026; development seed 1; `res/p2_m1_evid_*.json`)

`m1`, 200,000 lines, sums shown at gaps 0 / 4 / 8, last three readings, and what is closed at the end:

| alpha, p0 | Sums shown | The last reading | Closed at the end |
|---|---|---|---|
| 2, 0.2 | 52.9 / 45.0 / 41.7 | 78 / 64 / 56 | x y z (and `-`, which never occurs); w open |
| 2, 0.5 | 46.7 / 41.7 / 40.8 | 74 / 74 / 74 | w x y z, and the digits 0 and 2 |
| 4, 0.2 | 40.4 / 45.4 / 35.0 | 54 / 68 / 48 | y z |
| 4, 0.5 | 35.8 / 27.9 / 25.0 | 34 / 34 / 34 | w x y z, and the digits 0, 1, 2, 7 |

beside the first form at alpha 0.05 (37.5 / 40.0 / 22.5; z closed), `clock=none` (20.4 / 22.9 / 10.0) and `clock=oracle` (100).

**Q8 failed** by its letter: no setting ends with exactly the four letters closed (three letters and no digit at (2, 0.2); four letters and two or four digits where the starting chance is one half). **Q9 failed** (no setting at 90%; all are still rising at the end).

Reading (mine). The second form finds more letters than the first (three against one at the same length) and holds the digits open when entries start mostly open. Starting at one half is a trap: a digit that is closed half the time from birth is one the cells never come to use, and then closing it costs nothing and it stays closed. That is a failure of the rule to report, and it sets p0 = 0.2 for what follows. (2, 0.2) goes to B.

## B, as run (20 November 2026, written before its runs)

`g2`, development seed 1, 1,200,000 lines: `form=evid` (alpha 2, p0 0.2) as chosen on `m1`; the first form at alpha 0.05 beside it; and the two further settings the plan allows, both of the second form and both to be counted as tuned on this text if used: (alpha 4, p0 0.2), and (alpha 2, p0 0.2) with the evidence fading by 1/10,000 a period instead of 1/3,000.

## Outcome of B (entered 20 November 2026; development seed 1; `res/p2_g2_*.json`)

`g2`, 1,200,000 lines, never-shown pairs at gaps 0 / 4 / 8, mean of the last three readings; and the readings at 100,000 and 200,000 lines:

| Clock | Never-shown pairs | at 100,000 | at 200,000 | Closed (every reading from 100,000 lines) |
|---|---|---|---|---|
| second form, alpha 2, p0 0.2 (chosen on `m1`) | 95.1 / 95.1 / 95.1 | 32 | 87 | w x y z (and at two readings `-`, which never occurs) |
| first form, alpha 0.05 | 94.7 / 94.7 / 94.7 | 4 | 82 | w x y z |
| second form, alpha 4 (tuned on this text) | 94.3 / 94.3 / 94.3 | 66 | 85 | w x y z (once `-`) |
| second form, fading 1/10,000 (tuned on this text) | 95.1 / 95.1 / 95.1 | 25 | 89 | w x y z (once `-`) |
| `clock=none` (first plan) | 52.3 / 1.3 / 0.7 | | | nothing |
| `clock=oracle`, 400,000 lines (first plan) | 93.7 / 93.7 / 93.7 | | | w x y z, by the program |

**Q3 held** (the four letters are closed at the end, in all four runs, and no other byte value that occurs). The setting chosen on `m1` is kept for C; the two tuned on this text are not used further. With the letters closed the three gaps are the same lines to the chain, so the three figures are equal by construction.

Reading (mine). On this text the rule finds the letters within the first 100,000 lines, far sooner than I expected and far sooner than on `m1`. My account before the runs (the evidence must wait until the cells answer aligned lines) was therefore wrong for `g2`: the letters are closed before the sums are learned. I have not established what the early evidence is; what I can say is that the mismatch the entries read is over all the bytes of the line, not only the answer's, so a chain without the letters must already make the line's other bytes easier to name. This is to be looked at before anything is built on it.

## C, as run (20 November 2026, written before its runs)

`g2`, 1,200,000 lines, the five reported seeds, five arms at equal size and length: (1) the rule, second form, alpha 2, p0 0.2, fading 1/3,000 (chosen on `m1`, not tuned on `g2`); (2) the rule, first form, alpha 0.05 (18 numbers at the entries; added because it did as well in B); (3) `clock=none`; (4) the closing by chance, tendencies never changing, p0 0.2; (5) `clock=values:4`, four byte values closed by chance from the start. Arms (1) and (2) will be within ten points of each other if B is a guide, so both are also run on the seeds 1001 to 1010 and all fifteen reported.

## Outcome of C (entered 20 November 2026; `res/p2C_*.json`; the table is printed by `python3 table.py p2C <arms> <seeds>`)

`g2`, 1,200,000 lines, 256 cells, never-shown pairs, whole answer right, %, mean of the last three readings. Seeds in the order 73129, 93683, 116089, 406364, 545353.

| Arm | Gap of 0 letters: each seed; median | Gap of 4 | Gap of 8 | Closed at the end |
|---|---|---|---|---|
| the rule, second form (chosen on `m1`) | 96.7, 95.1, 96.9, 94.1, 97.4; **96.7** | the same | the same; **96.7** | w x y z on every seed |
| the rule, first form (alpha 0.05) | 95.7, 96.2, 97.1, 93.7, 95.9; **95.9** | the same | the same; **95.9** | w x y z on every seed |
| `clock=none` | 45.9, 54.7, 46.9, 54.5, 52.2; **52.2** | 1.3, 1.5, 1.5, 1.1, 0.8; **1.3** | 0.7, 1.0, 0.9, 0.5, 0.5; **0.7** | nothing |
| closing by chance, tendencies never changing | 4.9, 6.5, 5.9, 5.2, 4.7; **5.2** | 0.6, 0.9, 1.1, 0.6, 0.8; **0.8** | 0.7, 0.9, 0.8, 0.7, 0.5; **0.7** | nothing settled |
| four byte values closed by chance | 1.1, 3.3, 24.9, 0.3, 22.3; **3.3** | 0.7, 2.2, 15.7, 0.4, 14.2; **2.2** | 0.4, 1.0, 1.3, 0.2, 6.9; **1.0** | by the draw: 1 3 8 end; 0 2 7 z; 4 9 w x; 1 4 5 end; 8 + end w |

With the letters closed the three gaps are the same lines to the chain, so the rule's three figures are equal by construction. The two forms of the rule are within ten points of each other, so both were also run on the seeds 1001 to 1010. All fifteen seeds: second form 93.4 to 97.4, median 95.4 (the ten added: 95.2, 95.6, 94.4, 95.3, 94.9, 95.4, 97.1, 96.4, 93.4, 96.8); first form 93.7 to 97.1, median 95.9 (the ten added: 95.0, 96.1, 95.7, 95.8, 95.3, 96.0, 96.2, 97.1, 95.2, 97.0). The two forms cannot be told apart on this text. The rule is more than 40 points above every control at every gap on every seed, so no further seeds were added for those comparisons.

When the letters are found: exactly w x y z closed at every reading from 100,000 or 200,000 lines on in 28 of the 30 runs; from 400,000 in one (first form, 73129); and in one (second form, seed 1008) `=` is closed as well at the last reading, with 96.4% all the same. No digit is closed at the end of any of the thirty runs.

**H2 holds**: 96.7% at every gap at the median of the five reported seeds (least seed 94.1), where under 50% would refute; beside 52.2 / 1.3 / 0.7 with no clock. **Q4 held, Q5 held** (0.5 to 0.9 at the gap of 8), **Q6 held** (0.2 to 6.9), **Q7 held**.

**Reading (mine).**
1. A rule that names no byte, is told nothing about letters and reads no strength finds which byte values to keep out of the chain, on every one of fifteen seeds, within the first 100,000 to 200,000 lines, and the system then answers never-shown two-digit sums across 0 to 8 letters as it answers them with no letters (92.9% on `d2`, development seed). The holding cells of `loop_small_case` stood at 8 to 10% on this text after twice the lines.
2. What it costs in locality: one number sent out by the answering cell to all entries alike, and a common period of 256 bytes. Each entry's rule is otherwise its own state and its own two means.
3. The controls say the gain is the choice of bytes and not the passing over: the same closing by chance without learning leaves 5.2 / 0.8 / 0.7, below no clock at all, and four values closed by chance leave 3.3 / 2.2 / 1.0.
4. `=` closed on one seed is what I expected of a byte whose place in the chain does not matter to the answer (it is still on its delay-0 line when present): its evidence is near nothing and it can wander. It is a weakness to watch on texts where such a byte matters only sometimes.
5. What this does not show: that the rule serves where one closing alone earns nothing (`m1`: three letters of four at 200,000 lines), with many byte values (`PLAN_5.md`), or where the bytes to pass over are not a set of byte values at all (a clock for each byte value cannot pass over a digit in one place and keep it in another). Development level: one program, one text, fifteen seeds, nobody else's repetition.

**Cost.** 45 runs, 76 to 114 CPU-seconds each, about 75 CPU-minutes.

*Correction (20 November 2026, after looking at the readings for the report):* in the run of the second form on seed 1008 the `=` is closed at every reading from the first (100,000 lines) to the last, not only at the last as written above. The run's score is 96.4%, above its arm's median of 95.4%.
