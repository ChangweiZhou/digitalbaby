# nano/skip, eighth plan: three-digit numbers, the text that should part the two rules (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at d051d84, with the text `t3` added to `skip.py` and `expect.py` (`check_skip_output.txt` PASSED on 23 cases, two on this text). **Status.** Development level; no checker gate, seal or outside review is claimed. Written before any run on this text except the 300-line cases of the check; the outcomes of `PLAN_2.md` C and of `PLAN_4.md` to `PLAN_6.md` are not yet known.

**What this follows.** `PLAN_7.md`: the rule that passes over a byte which leaves the expectation of the next two bytes unchanged reaches 95.5% on `g2` (development seed), as the trial clock does (95.1%). With the next byte only, and the table counting across the line end, it passes over the first digit of every line and stays at 8.1%. My reading there: the rule tests what a byte does to the expectation of the next bytes and not whether the answer depends on it, and the two-byte form is saved on `g2` by the numbers having two digits.

**One change: the text `t3`.** Lines `aaa+<0 to 8 letters>bbb=cccc` with an end mark; four letter values; a and b from 0 to 999 at fixed width. 500 pairs are never shown (drawn by the seed); every other pair may be drawn, so almost every line is a pair seen for the first time. Scored on the 500 never-shown pairs with 0, 4 and 8 letters, whole answer (four digits) right, and by place; last three readings. The chain's nine lines reach every digit the answer needs when the letters are out of it.

**System.** As before, nothing retuned: 256 cells, 55,570 numbers that learn. Trial clock: second form, alpha 2, p0 0.2 (72 numbers at the entries). Expectation rule: cells for runs of up to 4 bytes, each counting the next byte and the byte after it, the table counting across the end mark, at least 20 counts, theta 0.2 (the first of the two values of `PLAN_7.md`; not retuned).

**Runs.** A (development seed 1, 1,200,000 lines): `clock=oracle`; `clock=none`; the trial clock; the expectation rule; and the expectation rule with nothing counted after the end mark (the stream's convention, which saved the first digit in `PLAN_7.md`). B (five reported seeds, 1,200,000 lines): `clock=oracle`, the trial clock, the expectation rule (table counting across the end mark).

**Question.** On numbers one digit longer than the expectation rule looks ahead, which rule still finds what to pass over?

**Hypothesis H8.** On `t3` (medians of five seeds, the gap of 8 letters): the expectation rule passes over the first digit of the first number on at least 90% of lines in the last 100,000 and has less than half of the oracle's whole answers right; the trial clock closes exactly the four letters and is within 15 points of the oracle. **Refuted if** the expectation rule has half the oracle's score or more, or the trial clock is more than 15 points below the oracle.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| W1 | `clock=oracle`, development seed: at least 50% of never-shown whole answers at 1,200,000 lines (if not, the comparison is read by place) | 0.50 |
| W2 | the expectation rule, table counting across the end mark: the first digit of the first number passed over on at least 90% of lines in the last 100,000 (development seed) | 0.80 |
| W3 | the same run: the answer's first two places are right less often than the oracle's by at least 20 points each | 0.65 |
| W4 | the trial clock, development seed: exactly w x y z closed at the end | 0.65 |
| W5 | H8 holds | 0.45 |
| W6 | the expectation rule with nothing counted after the end mark is within 15 points of the oracle (development seed) | 0.60 |

**Cost.** About 2.5 CPU-minutes a run; about 50 CPU-minutes. **Not run.** A look-ahead of three bytes (it would mend this text and fail at four digits by the same reasoning); longer chains; `clock=none` on the reported seeds.

## Outcome (entered 20 November 2026; `res/p8_t3_*.json`)

`t3` (three-digit numbers, 0 to 8 letters after the `+`), 1,200,000 lines, 500 never-shown pairs, whole answer (four digits) right, mean of the last three readings; one figure for the three gaps where the letters are out of the chain. Reported seeds in the order 73129, 93683, 116089, 406364, 545353.

| Clock | Each seed; median | Seed 1 | By place at the end, seed 1 (thousands, hundreds, tens, units) | Passed over in the last 100,000 lines |
|---|---|---|---|---|
| `clock=oracle` | 90.3, 91.2, 91.8, 94.5, 93.5; **91.8** | 94.3 | 99, 97, 97, 100 | the letters, by the program |
| the trial clock | 92.6, 94.8, 92.7, 94.3, 95.9; **94.3** | 93.6 | 99, 97, 98, 100 | w x y z closed at the end on every seed (from 100,000 to 500,000 lines); nothing else |
| the expectation rule, next two bytes, table counting across the end mark | 10.9, 8.0, 9.8, 8.3, 9.7; **9.7** | 8.3 | 72, 19, 96, 91 | every letter; **the first digit of the first number on every line (1.00, all six runs)**; the answer's second digit on 6 to 7% of lines |
| the same, nothing counted after the end mark | not run | 85.4 | 97, 97, 98, 92 | every letter; the answer's second digit on 7% |
| `clock=none` | not run | 13.3 / 0.1 / 0.1 (gaps 0 / 4 / 8) | 75, 20, 10, 11 at the gap of 8 | nothing |

**H8 holds.** The expectation rule passes over the first digit of the first number on every line and has 9.7% of whole answers at the median, about a tenth of the oracle's 91.8; the places that fail are the two that need that digit (72 and 19 where the oracle has 99 and 97), and the two that do not are intact. The trial clock closes exactly the four letters on every seed and is at 94.3%, level with the oracle (2.5 points above it at the median, which I read as nothing: the oracle's own seeds spread over 4 points). **W1, W2, W3, W4, W5, W6 held.**

**Reading (mine).**
1. The prediction made from `PLAN_7.md` before these runs is borne out: a rule that asks whether a byte changes the expectation of the next two bytes takes the first digit of a three-digit number for a byte like a letter. With nothing counted after the end mark that digit is again saved by the stream's convention (85.4%), which is no property of the rule; the same digit of the second number is saved by the letters or the `+` before it.
2. The expectation rule and the trial clock are not two ways to the same thing. The first finds bytes that are locally without consequence; the second finds bytes whose absence from the chain lowers the answering cell's mismatch. On `g2` these are the same bytes. Here they are not, and only the second is what the task needs.
3. The system itself, given an aligned chain, learns three-digit sums it was never shown to about 92 to 94%, though nearly every line it reads is a pair it has not read before (a million pairs, 1.2 million lines). That is a figure about the delay-chain system of `loop_small_case` that I have not seen measured there, and it is development level like the rest.
4. The trial clock finds the letters later here on some seeds (300,000 to 500,000 lines on three of six runs, against 100,000 to 200,000 on `g2`). Not examined.

**Cost.** About 50 CPU-minutes.
