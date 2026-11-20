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
