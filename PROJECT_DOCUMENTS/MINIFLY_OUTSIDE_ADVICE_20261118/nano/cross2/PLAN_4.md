# nano/cross2, fourth plan (E4): the same cells taught by the gradient through the whole line (a comparator, not something a cell has) (20 November 2026)

**Authorisation.** Mr. Xu's words of 20 November in `ROADMAP_SECOND_REVISION_20261120.md`, and, as relayed to me by the coordinating session in its message asking for this experiment, "Keep experimenting with the agents." (I have that sentence from the coordinating session only.) The experiment, its hypothesis and its thresholds are the coordinating session's; the roadmap's section 4 asks for a comparator by back-propagation where one exists. **Source.** Branch `claude/nano-cross2-8mmhfm` at 7e5aa97 plus the files of this commit (`kern6.c`, `fastk6.py`, `check_bp.py`, `hold_x.py` with `fast=1` and `adam=`). Written before any run at length; known to me: E2 and E3 above, the check's outputs, and three timings on seed 1 (g2, 40,000 lines at eta1 = 1/4 and 1: 3% of never-shown sums; 20,000 lines with Adam at 1e-3: 1%). Development seed 1; a small case; no checker gate.

**Question.** Can this network, at this size, learn g2 (and d2 without a delay line) when every cell strength is changed by the gradient through the whole line? If it can, the fault on these texts is in what the crossings leave out; if it cannot, the fault is not in the local rule alone.

**Learner, exactly.** The cells, sizes, readers and texts of E2 and E3 (256 cells, K = 16; g2 with `gap=8 chain=2`, 45,586 numbers that learn; d2 with no delay line, 23,054), 2,400,000 lines, read every 200,000. The answering cell and its return connections learn at every byte exactly as in `hold.py`. The cells' strengths change once at the end of each line by the instrument's pass backward over that line with unlimited crossings (`hold_x.py X=inf`, compiled in `kern6.c`), times eta1. Three things to be plain about: (1) the pass starts from the mismatch relayed over the return connections, which equal the answering cell's strengths transposed only after the return connections' random birth values have faded (they fade by 1e-5 a byte); with them equal, `check_bp.py` shows the pass is minus the derivative of the line's summed squared mismatch to 3e-9 against central differences; (2) the pass is cut at the first byte of the line (the lines are drawn independently); (3) one change a line, no averaging over lines.

**Check of the compiled pass** (`check_bp_output.txt`). Against `hold_x.py` at X=inf: agreement to 7e-15 over 300 lines at eta1 = 1/4 in six settings, and to 3e-13 on prefixes of 30 lines at eta1 = 1 and 10 lines at eta1 = 4. At eta1 = 1 and 4 the difference then grows with every line (g2: 5e-15, 2e-13, 6e-10, 2e-2 at 10, 30, 100, 300 lines with eta1 = 1): the learner amplifies differences of rounding size, so a long compiled run at those steps is one sample of what numpy would give from strengths differing in the last digit, not a copy of it. Controls (X=2) differ.

**Arms.** *Plain steps*: eta1 = 1/16, 1/4, 1, 4 (the two steps `hold.py` uses, and one on each side), on both texts, seed 1. *Adam* (a different learner, reported separately; **run only if no plain step reaches 20% of never-shown sums at the gap of 8 letters on g2**): the line's change for eta1 = 1 is given to Adam (0.9, 0.999, 1e-8), one step a line, step sizes 1e-4, 3e-4, 1e-3, 3e-3; the answering cell keeps its own rule. The five reported seeds are run only for an arm and step that passes 30% at the gap of 8 letters on g2 on seed 1 (the best step of that arm, beside nothing else).

**Measure.** As E2 and E3: whole answer right on 500 never-shown pairs (g2: at gaps 0, 4, 8), mean of the last three readings; sums shown; by place at the last reading; CPU.

**Hypothesis (the coordinating session's).** Taught by the gradient through the line, the network reaches at least 50% of never-shown sums on g2 at every gap within 2,400,000 lines. *Refuted if* it stays under 20% in both arms.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| D1 | g2, plain steps: the best step reaches at least 50% at every gap | 0.30 |
| D2 | g2, plain steps: the best step reaches at least 20% at the gap of 8 letters (so Adam is not run) | 0.50 |
| D3 | d2 without a delay line, plain steps: the best step reaches at least 20% of never-shown sums | 0.60 |
| D4 | g2: the best plain step is above the best forward arm of E2 (9.5% at the gap of 8 letters) by at least 5 points | 0.60 |
| D5 | at eta1 = 4 at least one of the two texts ends below its eta1 = 1 run (too large a step) | 0.70 |

**Cost.** About 12 CPU-minutes a run on g2 and 8 on d2; eight runs, about 80 CPU-minutes; Adam if run, four more; reported seeds if run, five more at one step.

**Not run.** Averaging over several lines; other forms of adaptive step; X=1 and X=2 at any time (numpy only, too slow for this length); more cells; a look inside.
