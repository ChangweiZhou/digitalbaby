# nano/skip, fourth plan: the trial clock where it should do nothing, where it was slow, and with its hand-set numbers moved (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at a6946a6; program and kernel unchanged. **Status.** Development level; no checker gate, seal or outside review is claimed. Written before its runs and before the outcome of `PLAN_2.md`, C, is known (those runs are in progress).

**What this follows.** `PLAN_2.md`: on `g2`, development seed, the trial clock closes the four letters within 100,000 lines and the system reaches 95% at every gap; on `m1` it had closed three letters at 200,000 lines (52.9 / 45.0 / 41.7% and rising), and with entries born half closed it closed digits as well. Three things are not known: whether the rule harms a text with nothing to pass over; whether it completes on `m1`; and how much rests on the numbers I set by hand.

**System.** As `PLAN_2.md`: 256 cells, 55,570 numbers that learn, and at the entries 72 (second form: alpha 2, p0 0.2, fading 1/3,000, P 256, floor 0.01) or 18 (first form, alpha 0.05). Nothing is retuned.

**Runs.**
A. `d2` (two-digit sums, no letters), five reported seeds, 400,000 lines: the rule (second form) beside `clock=none`.
B. `m1`, five reported seeds, 1,000,000 lines: the rule in both forms; `clock=none`; the closing by chance (alpha 0); `clock=oracle` (the check).
C. `g2`, development seed 1, 400,000 lines, second form, one hand-set number moved at a time: P 64; P 1,024; p0 0.1; p0 0.35; floor 0.05. Beside the unmoved setting's reading at 400,000 lines in `PLAN_2.md`, B (94).

**Question.** Is the rule safe where there is nothing to find, and does it finish on the text where it was slow?

**Hypothesis H4.** On `m1` the second form reaches at least 90% of the sums shown at every gap within 1,000,000 lines (median of five seeds, last three readings). **Refuted if** the median is under 90% at any gap.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| S1 | H4 holds | 0.50 |
| S2 | `d2`: the rule's median is within 3 points of `clock=none`, and on at least four of five seeds no byte value that occurs is closed at the end | 0.70 |
| S3 | `m1`: on at least one of the five seeds a digit is closed at the end (second form) | 0.30 |
| S4 | C: all five moved settings close exactly the four letters (of the bytes that occur) and have at least 85% at the last reading | 0.60 |
| S5 | `m1`: the first form's median is at least 90% at every gap | 0.35 |

**Cost.** About 35 CPU-minutes. **Not run.** `g1`; more than one number moved at a time; other texts.

## Outcome (entered 20 November 2026; `res/p4*.json`; tables by `python3 table.py p4A_d2 ...` and `p4B_m1 ...`)

Seeds in the order 73129, 93683, 116089, 406364, 545353; mean of the last three readings.

**A. `d2`, nothing to pass over, 400,000 lines, never-shown pairs.** The rule: 95.3, 95.3, 96.4, 93.5, 93.9; median **95.3**. `clock=none`: 96.6, 96.1, 95.5, 94.6, 95.5; median **95.5**. Closed at the end, of the byte values that occur: nothing on four seeds; `=` on one (116089, which has the rule's best score, 96.4). (Letter entries, whose bytes never occur here, sit closed on three seeds, which means nothing.) The rule costs 0.2 points at the median; each seed's run with the rule is within 2.2 points of its run without.

**B. `m1`, 1,000,000 lines, the 80 sums shown, gaps 0 / 4 / 8.**

| Arm | Gap 0: median (least) | Gap 4 | Gap 8 | Closed at the end |
|---|---|---|---|---|
| the rule, second form | **100** (100) | **100** (100) | **100** (100) | w x y z on every seed, at every reading from 400,000 lines |
| the rule, first form | 100 (99.2) | 100 (99.2) | 96.7 (86.2) | w x y z on four seeds (from 600,000 to 900,000 lines); w x y on one |
| `clock=none` | 98.8 (98.3) | 94.2 (88.8) | 10.8 (5.8) | nothing |
| closing by chance | 93.3 (83.8) | 88.3 (77.1) | 10.8 (8.3) | nothing settled |
| `clock=oracle` | 100 | 100 | 100 | by the program |

With a million lines the chain without a clock does learn the gaps of 0 and 4 letters on this text (the first number is then still within its nine lines) and not the gap of 8, where it is out of reach. The second form closes its first letters between 200,000 and 300,000 lines and all four by 400,000 on every seed; the first form is two to three times slower and on one seed has not closed the fourth letter.

**C. `g2`, development seed, 400,000 lines, one hand-set number moved** (unmoved: 94 at the last reading, letters closed from 100,000 lines). Last reading, and the reading from which exactly w x y z are closed: P 64: 88, 250,000 (`-`, which never occurs, comes and goes); P 1,024: 78 (92 the reading before), 100,000; p0 0.1: 89, 200,000; p0 0.35: 92, 50,000; floor 0.05: 83, 200,000. Every moved setting finds exactly the four letters and no digit; each is behind the unmoved one at this length.

**H4 holds** (100 at every gap on every seed). **S1 held. S2 held** (0.2 points; `=` closed on one seed of five). **S3 failed**: no digit is closed at the end on any seed (one digit was closed at one seed's first reading and reopened). **S4 failed**: all five find exactly the letters, but two are under 85 at the last reading (78 and 83). **S5 held** by its letter (medians 100, 100, 96.7), with one seed short of a letter.

**Reading (mine).**
1. Where there is nothing to pass over the rule does no measurable harm, apart from `=`, which it closes on one seed in five here as on one in fifteen on `g2`. `=` is the byte whose place in the chain the answer does not need. I count this as the rule's known loose end.
2. On the text where one closing alone earns nothing, the second form still arrives on every seed, about two to four times later than on `g2`. The trap of `PLAN_2.md` (digits closed) does not occur when entries are born mostly open.
3. The form that keeps two means is the more reliable of the two (five seeds of five against four), though the two could not be told apart on `g2`.
4. The hand-set numbers are not finely balanced as far as finding the letters goes; they do change how soon, by up to a factor of about two, and a higher floor costs score because a digit is then tried closed one period in twenty for ever.

**Cost.** About 30 CPU-minutes.
