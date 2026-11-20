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

## Outcome on the development seed (entered 20 November, before any reported seed was run)

Seed 1, 2,400,000 lines, plain steps (`res/hold_e4_*_1.json`; every reading in `report_e4_dev.txt`). Mean of the last three readings unless said.

**g2 (two delay lines), beside E2's forward arms:**

| | never-shown sums %, gap 0 / 4 / 8 | sums shown %, gap 0 / 4 / 8 | by place at the last reading | bytes right % | CPU s |
|---|---|---|---|---|---|
| forward, cross=1 (E2) | 7.8 / 7.9 / 7.5 | 9.3 / 8.9 / 8.7 | 97, 76, 16 | 47.1 | 1,548 |
| forward, cross=2 (E2) | 9.8 / 10.3 / 9.5 | 12.3 / 12.3 / 11.9 | 97, 78, 18 | 47.7 | 2,024 |
| gradient through the line, eta1 = 1/16 | 13.0 / 14.0 / 13.4 | 14.0 / 13.7 / 13.2 | 98, 83, 19 | 48.2 | 771 |
| gradient through the line, eta1 = 1/4 | 17.3 / 18.8 / 18.6 | 22.2 / 22.5 / 22.5 | 98, 85, 28 | 49.1 | 751 |
| **gradient through the line, eta1 = 1** | **69.7 / 68.4 / 68.6** | 72.8 / 73.9 / 73.9 | 99, 83, 96 | 53.6 | 767 |
| gradient through the line, eta1 = 4 | 64.4 / 64.7 / 64.1 | 68.5 / 70.3 / 70.1 | 98, 70, 98 | 53.1 | 782 |

Never-shown sums at the gap of 8 letters at each 200,000 lines (to lay beside other curves):

| lines, thousands | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eta1 = 1/16 | 3 | 7 | 5 | 7 | 9 | 8 | 9 | 11 | 12 | 12 | 14 | 14 |
| eta1 = 1/4 | 7 | 11 | 10 | 10 | 9 | 10 | 13 | 14 | 15 | 15 | 18 | 22 |
| eta1 = 1 | 8 | 11 | 12 | 8 | 13 | 13 | 11 | 15 | 48 | 66 | 61 | 79 |
| eta1 = 4 | 5 | 17 | 38 | 47 | 65 | 67 | 63 | 71 | 66 | 64 | 63 | 65 |
| forward cross=1 (E2) | 5 | 3 | 5 | 7 | 7 | 6 | 9 | 5 | 8 | 4 | 10 | 8 |
| forward cross=2 (E2) | 5 | 8 | 6 | 8 | 11 | 6 | 9 | 10 | 6 | 9 | 9 | 10 |

**d2 (no delay line), beside E3's forward arms** (never-shown sums; shown; by place; bytes right; CPU s): forward cross=1 2.3; 2.7; 89, 30, 9; 52.6; 746. Forward cross=2 8.0; 8.7; 97, 68, 16; 57.0; 999. Gradient through the line: eta1 = 1/16 11.1; 12.3; 99, 81, 18; 58.8; 452. **eta1 = 1/4 69.0**; 71.5; 98, 85, 92; 67.3; 448 (readings 6, 28, 33, 37, 38, 10, 16, 32, 54, 61, 71, 76: a fall at 1,200,000 lines and a recovery). eta1 = 1 55.8; 55.9; 97, 61, 97; 64.9; 445 (6, 8, 18, 35, 39, 42, 63, 58, 49, 51, 61, 55). eta1 = 4 6.0; 6.5; 91, 27, 28; 54.3; 447.

**The hypothesis is not refuted, and its positive form holds on this seed**: at eta1 = 1 the gradient through the line gives 68 to 70% of never-shown sums on g2 at every gap, where 50% was asked. A plain step passes 20%, so **the Adam arm was not run**. Two steps pass 30% at the gap of 8 letters; the plan names the best, which by its measure is eta1 = 1 (68.6 against 64.1), and the five reported seeds are run at that step. (eta1 = 4 rises much sooner, from 400,000 lines, and levels near 65%; it is not run on the reported seeds, as the plan names one step.)

**Scores.** D1 held (69.7 / 68.4 / 68.6). D2 held. D3 held (69.0 at eta1 = 1/4). D4 held (68.6 against 9.5). D5 held (g2: 64.4 against 68.9; d2: 6.0 against 55.8).

**What it shows and does not** (my reading; one seed).
- The network of E2 and E3, at the same size and length, learns both texts when its cells are taught by the gradient through the whole line: 69% on each at the best step, against 8 to 10% (g2) and 2 to 8% (d2) with the forward rules. So within 2,400,000 lines the difference lies in what the forward rules leave out and not in the size or form of the network.
- **The step matters as much as the rule.** On g2 the two smaller steps end at 13 and 18%, not far above the forward arms; on d2 the largest ends at 6%. The forward arms were run at eta1 = 1 only, so "forward rule against gradient" is compared at one step for the forward rule and at the best of four for the gradient.
- **The level stretch is there with the gradient too, at eta1 = 1**: 8 to 15% for 1,600,000 lines, then 48% at 1,800,000 and 79% at 2,400,000. The coordinating session reports that its one-crossing run of this seed, continued to 9,600,000 lines, was level for about five million lines and then rose to 46% (its figures, `loop_small_case/PLAN_12.md` at 59f2f4342, which I have not read). Laid side by side at this step, the gradient through the line leaves the level stretch after about 1.6 million lines where one crossing needs about five to six million: sooner by a factor of three to four. At eta1 = 4 the gradient shows no level stretch (17% at 400,000, 65% at 1,000,000), so the stretch is not a property of the network and text alone; it depends on the rule and the step together. Whether one crossing at eta1 = 4 would also shorten its stretch was not run.
- On g2 the units digit is what the gradient adds (96 to 98% against 16 to 18%); the tens digit stays at 70 to 83%, and that is what holds the whole answer at 65 to 79%.
- In CPU time the pass backward is cheaper here than the forward rules: about 770 s a run on g2 against 1,548 and 2,024, because the forward program updates a trace for every strength at every byte. It needs the line's states stored, which a cell does not have.
