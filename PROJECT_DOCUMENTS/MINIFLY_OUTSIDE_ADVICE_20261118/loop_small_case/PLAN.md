# The loop at the smallest size: cells in between that learn from a relayed mismatch (plan, fixed before any run)

**Date:** 19 November 2026. **By** the outside session that wrote the audits beside this folder.

**Authorisation.** Mr. Xu, 19 November 2026, to this session, after the audit of the tissue's programs: "So the current agents are obviously stuck. If you have more thoughts, try to implement them to unblock the current situation". This plan implements points 1 to 3 of section 6 of `AUDIT_OF_THE_TISSUE_PROGRAMS_20261119.md` and nothing else. It is a small case in the sense of the reviewer's status rows 137 to 153. It is not a version of the adopted core, takes no version number, and claims no checker gate.

**Source.** This branch (`claude/friendly-darwin-8mmhfm`) at 18c4df20; the text and the score are those of the reviewer's `yesno/addtext.py` (branch `claude/vibrant-cerf-sje726` at 21246a44), written again here.

## 1. The question, the hypothesis and what would refute it

The audit found that in every system of 18 and 19 November nothing between the input and the answering cell learned from an error, and that the systems which answered sums never shown had the adding laid down by the program. The question here: **if the cells in between change their own strengths from a mismatch that is formed at the answering cell and relayed to them over return connections, with nothing about digits or addition laid down, do they come to answer sums never shown?**

*Hypothesis.* They do: such a system learns the lines it is shown and answers a majority of the sums never shown, and the order of the digits appears in the learned strengths.

*Refuted if* sums never shown stay at the level of a table and of guessing (15% or below at the median over the seeds) while sums shown are learned.

## 2. The text and the score (the reviewer's, unchanged)

Lines `a+b=c` followed by an end mark, a and b from 0 to 9, read one byte at a time. Of the 100 pairs, 20 are never shown (drawn by the seed exactly as in `addtext.py`). Learning: 60,000 lines drawn with replacement from the 80 pairs shown, one stream, one byte at a time, no batch, nothing reset, nothing averaged over lines. At every byte but the end mark the system names the next byte before it arrives.

Score, with learning off and no random choice, every pair once: the *whole answer* is right when every byte after `=` up to the end mark is named right (the reviewer's definition); also given with the end mark included. Reported apart for the 80 pairs shown and the 20 never shown, and for the never-shown pairs whose sum occurs in no pair shown.

## 3. The system

Laid down by the program, and nothing else: one line for each byte value; four delay cells in a chain, each passing on at the next byte the byte it held (so the present byte and the four before it are present as 5 x 13 lines; the reviewer's and the workers' systems have the same or more); the sizes; the starting strengths drawn at random.

- **H cells in between.** Cell j has a strength for each of the 65 lines and a threshold. Its drive is the sum of its strengths from the five active lines less its threshold; its output h_j is the logistic function of the drive, between 0 and 1.
- **One answering cell** with one branch for each byte value in use (the construction of the reviewer's rows 137 to 153). Branch v gives a_v, the sum of its strengths from the cells in between plus a constant. The cell names the byte of its largest branch. It reads the cells in between and nothing else.
- **The mismatch** is formed in the answering cell when the next byte arrives: m_v = 1 - a_v for the byte that arrived and -a_v for every other (the reviewer's rule told the right byte; Mr. Xu, `WHAT_A_NEURON_HAS` section 7: the right answer shown as input after the attempt is the form of teaching). Under **yes or no** (rule Q): m_e = ok - a_e for the byte named; on yes the others get -a_v; on no the others get nothing.
- **The answering cell changes** each branch's strengths by eta2 m_v h_j / n, n = |h|^2 + 1.
- **The relay.** Cell j in between receives r_j, the sum over branches of R_jv m_v, over return connections R. It changes its strengths from the five active lines, and its threshold, by eta1 r_j h_j (1 - h_j). What it uses: the signal that reaches it, its own output, and which of its lines are active.
- **The return connections,** three arms: fixed at their random birth values; changed by the same product as the forward strength between the same two ends (R_jv by eta2 m_v h_j / n), with both decaying at rate lambda, so that the two come to agree without one being copied from the other; and, as the comparator, replaced at every step by the answering cell's own forward strengths, which is back-propagation through this network and is not something a cell has.

No pass runs backward over the line; one relay a byte; no cell is handed the answer except the answering cell, to which it arrives as the next byte.

**What is still not as a cell has it,** stated before any run: a strength may be of either sign and may change sign; the mismatch is signed; the answering cell has a branch for each byte value; every cell in between reads all 65 lines; the delay chain is given. Each is to be removed one at a time under the same score if the hypothesis holds.

## 4. Arms

| Arm | Cells in between | Return connections |
|---|---|---|
| `none` | none: the answering cell reads the 65 lines | - |
| `fixed` | H, strengths never changing | - |
| `relay-fixed` | H, learning | fixed random |
| `relay-learned` | H, learning | learned (the candidate) |
| `comparator` | H, learning | the forward strengths (back-propagation) |
| `relay-learned, yes or no` | H, learning | learned; mismatch from yes or no only |

## 5. Numbers, and how they are chosen

On one development seed (1, the default of `addtext.py`), which is not among the seeds reported: H in 8, 16, 32, 64; eta1 in 1/16, 1/4, 1; eta2 in 1/16, 1/4; lambda in 0, 1e-5, 1e-4. The setting is chosen for `relay-learned` by sums never shown on that seed (ties by sums shown), written into section 8 below before the reported seeds are run, and used for every arm. If no setting learns the sums shown on the development seed, that is reported as the result, and the fallbacks tried are named.

**Reported seeds,** fixed now: 73129, 93683, 116089, 406364, 545353 (the five that the lines used on 19 November), each run once in every arm with the stream of `addtext.py` for that seed.

## 6. Success, failure, and predictions (mine, before any run)

*Holds:* `relay-learned` has the whole answer right on at least 90% of sums shown and at least 50% of sums never shown, at the median over the five seeds. *Refuted:* 15% or below on sums never shown at the median, with sums shown learned. Anything between is reported as partial.

| | Statement | Chance |
|---|---|---|
| P1 | The comparator reaches 90% on sums shown on at least 4 of 5 seeds | 0.85 |
| P2 | The comparator reaches 50% on sums never shown at the median | 0.60 |
| P3 | `relay-learned` is within 15 points of the comparator on sums shown at the median | 0.70 |
| P4 | `relay-learned` reaches 50% on sums never shown at the median | 0.50 |
| P5 | `fixed` stays at or below 15% on sums never shown at the median | 0.80 |
| P6 | `relay-fixed` reaches 70% on sums shown at the median | 0.60 |
| P7 | `relay-learned` under yes or no reaches 50% on sums never shown at the median | 0.35 |
| P8 | Where a network has at least 50% of never-shown sums right, at least three of its cells in between have strengths from the ten digit lines in the order of the digits (correlation at least 0.9 in size) at both delays of a pair | 0.55 |

## 7. Cost, and what is not run

A run is about 330,000 bytes on at most 64 cells; seconds of one core. The whole plan is a few CPU-minutes. Not run: numbers above 9; the stream of step 1; any tissue; more than one answering cell; a second task; anything about keeping.

## 8. The setting chosen on the development seed

Entered on 19 November after the development runs and before any reported seed was run.

The grid of section 5 on seed 1, arm `relay-learned` (72 runs; `python3 cells_loop.py dev`): sums shown are learned (100% with 32 or 64 cells in between at most settings), and sums never shown are between 0% and 10% at every one of the 72 settings. By the rule of section 5 the setting is the first of the two best: **H = 32, eta1 = 1/16, eta2 = 1/4, lambda = 0** (shown 100.0, never shown 10.0).

Seen on development seeds 1, 2 and 3 before the reported seeds were run, and stated here so that the report cannot be read as a surprise: the comparator gives the same picture as the local rule (shown 97.5 to 100, never shown 0 to 10 at 60,000 lines; the same at 240,000 lines; a fading of the cells' strengths at 1e-5 a byte changes nothing and at 1e-4 prevents learning). The option for that fading (`lam1`) was added to the runner for these runs; it is off in the reported runs. My expectation for the reported seeds is therefore that the hypothesis is refuted in this form for the local rule and for the comparator alike, and that P2 and P4 fail.
