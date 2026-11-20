# Sixth plan: one crossing between cells, forward in time (roadmap R3, second stage) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Keep working, see how much you can push along the road map. Feel free to revise anything you like." **Source.** This branch at bb75de2e (the fifth plan and `hold.py`). Written after the fifth plan's development runs and before any run of what is described here. Development level; a small case; not a version of the adopted core; no checker gate.

## 1. Where the fifth plan stopped

Cells that hold bytes themselves (`hold.py`: 256 gated cells, each reading the present byte and 16 other cells one byte late, a trace at each connection running forward, no delay chain) learn single-digit sums shown (89% to 92% on the development seed against 2.5% for cells that never change) and do not learn two-digit sums (1% to 2% of never-shown sums at 600,000 lines; by place: hundreds 86 to 87, tens 22 to 26, units 7 to 12).

*My reading.* In that system a cell changes only for what its own state gives the answering cell directly (no crossing between cells: the form the audit of 18 November called X = 0). The units digit of a sum needs two cells' worth of holding (a digit of each operand) and then a cell that reads both. The reading cell is paid for what it does; the holding cells are not, because one operand digit alone says nothing about (a + b) mod 10. Only the hundreds digit depends on a held digit by itself, and only it is learned.

## 2. The question, the hypothesis, what would refute it

**If a cell that reads another cell passes on to it what it received from the answering cell, times its own sensitivity to that connection, do two-digit sums become learnable without the delay chain and without any pass backward in time?**

*Hypothesis.* Yes. One crossing, taken at the last step, is enough for the holding cells to be taught by the cells that read them.

*Refuted if* the system with the crossing stays at or below 15% of never-shown sums at 600,000 lines on the development seed. If so, the instrument of section 4 says which of three things is the case: one crossing is not enough but more are (the instrument learns at larger X); the timing of the crossing matters (X = 1 at any time learns, at the last step does not); or these cells cannot learn the text in this time under any credit (the instrument fails at X = inf).

## 3. The system

`hold.py` as in the fifth plan, with one addition (`cross=1`). At each byte, after the mismatch is formed and relayed as before (cell k receives r_k):

- reader k sends to each cell j that it reads the quantity r_k times (k's sensitivity of its state to its candidate, times the return strength for the candidate, plus k's sensitivity to its gate, times the return strength for the gate). Both sensitivities are k's own quantities at this byte;
- cell j adds up what its readers send, and changes each of its strengths by that sum times the strength's trace **as it stood one byte earlier** (the state that k read was j's state one byte earlier). A cell keeps that one earlier value of each trace;
- this is in addition to the change of the fifth plan (r_j times the present trace).

Nothing runs backward over the line. What a cell uses: what reaches it over its connections, its own state and sensitivities, its own traces.

**Return strengths, two arms.** `ret=exact`: the return strength of a connection is its forward strength, the same connection used from its other end. Whether a cell may use the strength of a connection it sends over is not mine to settle. Mr. Xu's ruling recorded in `WHAT_A_NEURON_HAS_20261118.md`, section 16 (the reviewer's translation: "The strength of a connection between cells is determined jointly by the two ends of the connection; it is a local quantity") may be read as allowing it; that reading is mine, and the second arm does not depend on it. `ret=learned`: a separate return strength at each connection, starting at an independent random value, receiving the same changes as the forward strength, both fading at 1e-5 a byte, so that the two come to agree without one being copied (the construction used for the answering cell's return connections since the first plan). The second is the candidate; the first is its reference.

**Sizes.** 256 cells; 14 lines; K = 16. Strengths that learn: 7,168 from lines, 8,192 between cells, 512 thresholds, 3,598 at the answering cell: 19,470; return strengths 3,584 (answering cell) and 8,192 (between cells, `ret=learned`). One trace and one earlier trace for each of the cells' 15,872 strengths. State: 256 numbers.

## 4. An instrument (not something a cell has)

`hold_x.py`: the same cells, with every cell's change computed by a pass backward over the line just read, in which a path may cross from one cell to another at most X times: X = 0 (own path only; what the traces of the fifth plan give), X = 1L (one crossing, at the last step; what `cross=1 ret=exact` gives), X = 1 (one crossing at any time), X = 2, X = inf (back-propagation through the line: the comparator the fifth plan lacked). The answering cell learns at every byte as before; the cells change once at the end of each line. It is used to measure how many crossings the text needs and to check the forward program against (`python3 hold_x.py check`: at fixed strengths the forward program's accumulated changes over a line must equal the instrument's at X = 0 and at X = 1L).

## 5. Runs

Development seed 1, two-digit sums, 600,000 lines, readings every 100,000, eta1 = 1/4 (and 1 where 1/4 is still rising or flat): `cross=1 ret=exact`; `cross=1 ret=learned`; instrument at X = 0, 1L, 1, 2, inf. If a forward arm reaches 30% of never-shown sums, it is run with `cross=0` beside it on the five reported seeds (73129, 93683, 116089, 406364, 545353), at the setting chosen on the development seed and entered below before those runs.

Score as in the fifth plan: whole answer right on 500 never-shown pairs and on 500 shown pairs, mean of the last three readings; by place at the end.

## 6. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| K1 | `cross=1 ret=exact` has at least 30% of never-shown sums right on the development seed | 0.50 |
| K2 | `cross=1 ret=learned` has at least 30% | 0.35 |
| K3 | with `cross=1 ret=exact` the units digit is right on at least 50% of never-shown sums (it was 7 to 12) | 0.60 |
| K4 | the instrument at X = 0 stays at or below 10% | 0.75 |
| K5 | the instrument at X = inf reaches at least 50% | 0.60 |
| K6 | the instrument at X = 1 (any time) reaches at least 30% | 0.50 |

## 7. Cost, and what is not run

A forward run is about 15 to 20 CPU-minutes, an instrument run about 25 to 40. Development: about nine runs, three to four CPU-hours. Five reported seeds, two arms: about three CPU-hours. Not run: more than one crossing in the forward program; cells that hold over a gap of varying length (letters between the operands, as in the first family); subtraction or keeping in this system; any size other than 256 cells.
