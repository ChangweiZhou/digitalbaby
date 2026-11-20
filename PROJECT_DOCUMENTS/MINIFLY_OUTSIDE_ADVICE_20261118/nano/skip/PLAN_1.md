# nano/skip, first plan: a clock for the delay chain; the program, its check, and whether a sum kept at the lines separates the letters (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." Earlier the same day (`ROADMAP_SECOND_REVISION_20261120.md`, section 1): "Also the network could have many different types of cells that adjusts around different time clocks, having different memory ability, ability to output signals (chemicals) etc. But we need to make sure not testing everything at once and leave a mess."
**Source.** `claude/friendly-darwin-8mmhfm` at 9346c104e; this line's branch is `claude/nano-skip-8mmhfm`; files changed only in `nano/skip/`. **Status.** Development level, development seed 1 only in this plan; no checker gate, seal or outside review is claimed. Written before any run of an experiment; the check of the compiled kernel against numpy (`check_skip_output.txt`, 300 lines a case, PASSED) was run before this plan, being a test of the program and not of the question.

## 1. What this starts from

(a) `loop_small_case`: the delay-chain system learns `aa+bb=ccc` to 84% of never-shown pairs (median of five seeds, 400,000 lines). (b) With 0 to 8 random letters between the `+` and `bb` (text `g2`) gated holding cells reach 8 to 10% after 2,400,000 lines (`PLAN_11.md`, `PLAN_12.md`). The coordinator's idea, which this line is to improve or refute: let the chain step only for bytes that the cells listen to, and find those bytes by a local rule.

## 2. The system (copied from `loop_small_case/cells2.py`, plain options; the texts and the scoring from `hold.py`)

18 byte values (digits, `+`, `=`, the end mark, `-` which does not occur, and the letters w x y z). For each byte value one line that carries the present byte (delay 0) and nine chain lines (delays 1 to 9): 180 lines. 256 logistic cells in between, each reading all the lines; one answering cell with 18 branches; the mismatch relayed back over return strengths that receive the forward strengths' changes; steps 1/4 and 1/16, fading 1e-5 of the answering cell's and the return strengths. **Numbers that learn: 55,570** (46,080 strengths from lines, 256 thresholds, 4,608 + 18 at the answering cell, 4,608 return strengths). One stream, a byte at a time; the system names every next byte within a line.

*The one change: a clock.* When a byte arrives, the byte that was present either enters the chain (which then moves on by one) or is passed over and is on no line any more. The present byte is always on its delay-0 line. `clock=none`: every byte enters (the system of (a) on this text). `clock=oracle`: the letters are passed over; the program is told which bytes are letters, so this is a check of the program and nothing else. With the oracle the chain holds `aa+bb=ccc` at the moment of every answer; it differs from (a) only in that the system also names the bytes that follow the `+` and the letters.

*A second change, measured separately:* `lam1`, a fading of a cell's strengths from the lines that are active at that byte (0 in `cells2.py`).

*The quantity measured.* For each byte value v: Q(v), the sum over v's nine chain lines and over the 256 cells of the sizes of the strengths the cells hold from those lines. It lives at the lines (each line is taken to know the sum of the sizes of the strengths held from it, as an edge decided by both ends would; the nine chain lines of one byte value pool their sums). Q0(v) is its value at birth. The rule of the next plan would let v enter the chain only while Q(v) is at least theta times Q0(v); it names no byte and reads nothing else.

*Scoring.* As `hold.py`: 500 never-shown pairs at gaps of 0, 4 and 8 letters, the letters fixed by the pair, whole answer right; each line read from the state the stream left; mean of the last three readings. On `m1` (single digits, `a+<letters>b=c` with c the sum modulo 10): the 80 sums shown, as in `hold.py`.

## 3. The question, the hypothesis, what would refute it

**Question.** In the chain without a clock, where nothing is aligned yet, does Q(v)/Q0(v) come to separate the four letters from the 13 other byte values that occur, and by when?

*What I expect and why (Reading, mine).* Without fading (`lam1=0`) a strength from a line that predicts nothing still moves at every byte at which the line is active, by the relayed mismatch, so its size grows like a random walk and Q need not be small for the letters. With a fading of the strengths in use, a strength that is not pushed consistently falls to a floor set by the noise, and one that is pushed consistently is held up. Against this: in the unaligned chain the letters on the chain lines do say how long the gap is, and the cells could use that; and the digits of the first number are at fixed delays only when the gap happens to have one length.

**Hypothesis H1.** For at least one `lam1` of 1e-4, 1e-3, 1e-2, at 400,000 lines of `g2` with `clock=none`, each of the four letters has Q/Q0 at most 0.8 times the smallest Q/Q0 among the 13 other byte values that occur. **Refuted if** this holds for no `lam1` of the three.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| P1 | `clock=oracle`, `lam1=0`, `g2`, 400,000 lines: at least 75% of never-shown pairs at every gap (the check that the program is the system of (a)) | 0.80 |
| P2 | `clock=none`, `lam1=0`, `g2`, 1,200,000 lines: under 20% at the gap of 8 letters | 0.85 |
| P3 | `lam1=0`: the condition of H1 fails (no separation without fading) | 0.70 |
| P4 | H1 holds | 0.45 |
| P5 | `clock=oracle` with `lam1=1e-3` is within 10 points of `clock=oracle` with `lam1=0` (the fading costs little) | 0.50 |
| P6 | `m1`, `clock=oracle`, `lam1=0`, 200,000 lines: at least 90% of the sums shown at every gap | 0.75 |

## 4. Runs (development seed 1, compiled kernel)

1. `g2`, 400,000 lines: `clock=oracle` and `clock=none`, each with `lam1` 0, 1e-4, 1e-3, 1e-2 (eight runs). 2. `g2`, 1,200,000 lines, `clock=none`, `lam1=0` (the control for the next plan). 3. `d2` (no letters), 400,000 lines, `clock=none`: the system of (a) in this program. 4. `m1`, 200,000 lines: `clock=oracle` and `clock=none` with `lam1` 0, 1e-3, 1e-2 (six runs).

**Cost.** Estimated from `kern3.c` (16 to 34 s for 400,000 lines of ten bytes): about 1 CPU-minute for 400,000 lines of `g2`; about 15 CPU-minutes in all.

**Not run.** The rule itself; the reported seeds; any other quantity than Q; more or fewer than nine chain lines; more cells.

## Outcome (entered 20 November 2026 after the runs; development seed 1; result files `res/p1_*.json`)

Never-shown pairs, whole answer right, %, mean of the last three readings, at gaps of 0 / 4 / 8 letters (on `m1`: the sums shown). Every figure of `clock=oracle` is beside `clock=none` at the same size and length.

| Text, lines | `lam1` | `clock=none` | `clock=oracle` |
|---|---|---|---|
| `d2` (no letters), 400,000 | 0 | 92.9 | (the same text) |
| `g2`, 400,000 | 0 | 19.7 / 1.5 / 0.9 | 93.7 / 93.7 / 93.7 |
| `g2`, 400,000 | 1e-4 | 1.0 / 0.7 / 0.5 | 58.5 / 58.5 / 58.5 |
| `g2`, 400,000 | 1e-3 | 0.7 / 1.1 / 0.7 | 1.9 / 1.9 / 1.9 |
| `g2`, 400,000 | 1e-2 | 0.9 / 0.7 / 0.9 | 0.9 / 0.9 / 0.9 |
| `g2`, 1,200,000 | 0 | 52.3 / 1.3 / 0.7 | not run |
| `m1`, 200,000 | 0 | 20.4 / 22.9 / 10.0 | 100 / 100 / 100 |
| `m1`, 200,000 | 1e-3 | 5.0 / 9.2 / 11.2 | 42.5 / 42.5 / 42.5 |
| `m1`, 200,000 | 1e-2 | 1.7 / 9.2 / 10.4 | 12.1 / 12.1 / 12.1 |

Q(v)/Q0(v) at the end of the `g2` runs with `clock=none` (range over the byte values of each kind):

| `lam1` | ten digits | `+` | `=` | end mark | four letters |
|---|---|---|---|---|---|
| 0, 400,000 lines | 2.36 to 2.99 | 2.06 | 3.83 | 2.75 | 2.72 to 2.80 |
| 0, 1,200,000 lines | 2.72 to 3.45 | 2.38 | 4.40 | 3.18 | 2.87 to 2.97 |
| 1e-4 | 0.06 to 0.10 | 0.04 | 0.19 | 0.06 | 0.07 |
| 1e-3 | 0.02 to 0.03 | 0.02 | 0.15 | 0.02 | 0.03 |
| 1e-2 | 0.00 to 0.01 | 0.00 | 0.13 | 0.01 | 0.01 |

The same sum line by line, over birth, `g2`, `clock=none`, `lam1=0`, 400,000 lines, mean over the byte values of a kind, at delays 0 to 9:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| digits | 2.82 | 2.82 | 3.32 | 3.40 | 1.99 | 2.21 | 2.91 | 2.52 | 2.21 | 2.28 |
| letters | 4.67 | 5.68 | 3.93 | 2.64 | 2.25 | 2.29 | 2.04 | 1.91 | 2.13 | 1.98 |

and with `clock=oracle` (the chain aligned): digits 2.32, 1.41, 3.84, 8.86, 1.11, 3.66, 8.81, 1.25, 1.17, 1.78; letters 6.12 at delay 0 and 0.96 to 1.03 (never active) elsewhere.

**The program is the system of (a).** With the oracle clock the three gaps give the same figure by construction, 93.7% on this seed, beside 92.9% for the same program on the text without letters. (The report's 84% is a median of five other seeds with `cells2.py`'s scoring, which fills the chain with end marks; here each scored line is read from the state the stream left.)

**H1 is refuted.** At no `lam1` are the letters' sums below the other bytes' by the margin asked, or at all: without fading the letters lie inside the digits' range; with fading every sum falls together (to 2 to 10% of birth) and the system stops learning even with the oracle clock (58.5% at 1e-4, 1.9% at 1e-3).

**Where it fails (Reading, mine).** The quantity does not separate the bytes, and for a reason that no threshold would mend. (i) In the chain without a clock the letters on the first chain lines are the most listened to of all (5.68 and 3.93 times birth at delays 1 and 2, against 2.82 and 3.32 for the digits): a letter one or two bytes back tells the cells that the two digits now arriving are the second number and that `=` comes next. The cells listen to the letters for where the line is, though no letter says anything about the answer. A sum of sizes cannot tell the first use from the second. (ii) At the longer delays, where neither kind is of use, both kinds stand at about twice birth: a strength that is pushed by the relayed mismatch without consistency still wanders away from its birth value. (iii) A fading strong enough to pull such a strength back removes the learning first.

The measure that does tell the two apart is what happens to the mismatch when a byte is left out of the chain, and that cannot be read from the strengths held while the byte is in the chain. The next plan lets each byte value try it.

**Predictions.** P1 held (93.7 at every gap). P2 held (0.7 at the gap of 8). P3 held (no separation without fading). P4 failed (H1 refuted). P5 failed (1.9 against 93.7: the fading costs nearly everything at 1e-3). P6 held (100).

**Cost.** 82.7 CPU-seconds for the longest run, about 6 CPU-minutes in all.
