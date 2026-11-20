# nano/loop, first plan (roadmap step L1): the answering cell, least squares against branches that compete (20 November 2026)

**Authorisation.** `ROADMAP_SECOND_REVISION_20261120.md`, sections 1 and 6 (line "L1, then L2", folder `nano/loop`), given to this session by the coordinating session. **Source.** Branch `claude/nano-loop-8mmhfm`, started from commit fe397ca. Written before any run of the step. What was run before this plan: `check.py` (3,000 bytes a case), `loop.py tables`, and two runs of one reading each to measure the cost (256 and 1,024 cells, arm P, eta1 = 1, eta2 = 1, return strengths at the earlier size: 2.0975 and 1.9392 nats); these two are not used for any choice or claim. Development level; no checker gate, seal or review is claimed.

## Copied, and from where

From `loop_small_case` at commit fe397ca, into this folder and changed only here: `kern3.c` -> `kernp.c` (the changes are listed at its head); the class `Sys` of `cells2.py` (plain system only), the ctypes wrapper of `fastk3.py` and the scoring and tables of `prose.py` -> `loop.py`; the manner of `check_fast.py` -> `check.py`. Nothing outside `nano/loop/` is changed.

## The test

`nano/data/tinyshakespeare.txt` with the split of `nano/data/README.md`: bytes 0 to 1,003,853 are the stream, the last 111,540 bytes are never read by the learner. 65 byte values, each given its own line at each delay; no other preparation. The stream is read 8 times, one byte at a time, each next byte named before it arrives; nothing is reset between readings. **Measure:** after the last reading, with learning off, every one of the 111,540 bytes of the last tenth is scored from the D bytes before it (for its first bytes these are the last bytes of the stream): mean loss in nats and in bits a byte, and the share of bytes named right (largest branch).

## The exact learner

`loop.Sys` (numpy reference `Sys.step`; compiled `kernp.c`; `check.py` compares them: largest difference over largest strength 8e-16 to 6e-13 on 13 cases, `check_output.txt`). D = 10 delays, so 650 lines. H logistic cells, each reading every line (strengths born uniform in -1/2..1/2, threshold 0). One answering cell with 65 branches (strengths and constants born at 0). Step of the answering cell: eta2 / (h.h + 1) times mismatch times the sending cell's output. A cell in between receives the mismatches over its return strengths and changes its strengths from the 10 active lines and its threshold by eta1 times that sum times h(1 - h). A return strength receives the same change as the forward strength between the same ends; both fade by 1e-5 a byte. Return strengths are born uniform in -1..1 times `rscale`.

- **Arm LS** (as before): mismatch = indicator of the byte that arrived, less the branch outputs a. Its loss is **crude**: the outputs are cut off below at 1e-3 and divided by their sum; this is not a probability the cell was taught to give.
- **Arm P**: p = exp(a) / sum of exp(a); mismatch = the same indicator less p; loss = -log p of the byte that arrived. Everything else identical, the normalised step included.

Numbers that learn: H x 650 strengths from lines, H thresholds, 65 x H + 65 in the answering cell, 65 x H return strengths: **199,745 with 256 cells, 798,785 with 1,024**. (With return strengths born at nothing the return strengths equal the forward ones throughout: 183,105 and 732,225 distinct numbers.)

## Question, hypothesis, refutation

**Question.** At equal size and length, does an answering cell whose branches compete give a lower loss on text never read than the least-squares one?

**Hypothesis.** Yes at both sizes. **Refuted if**, on the mean of the claim seeds, arm P's loss is not lower than arm LS's crude loss at either size, or arm P names fewer bytes right than arm LS by more than 1 point at both sizes.

## Choosing the settings (development seed 1 only)

For each arm separately, by the loss on the last tenth:
1. 256 cells, 2 readings, the full grid: eta1 in {1/4, 1, 4}; eta2 in {1/64, 1/16, 1/4, 1} for LS and {1/4, 1, 4, 16} for P; rscale in {0, 0.03, 0.3, 1}. 96 runs.
2. 256 cells, 8 readings: each arm's three best settings of stage 1, and for each of them the same eta1, eta2 with rscale = 0 if not already among them. The best at 8 readings is the arm's setting; the best with rscale = 0 is the setting of the arm "born at nothing".
3. 1,024 cells, 4 readings: the arm's setting and the same with eta2 four times smaller and four times larger (6 runs). If a neighbour is better by more than 0.01 nats it is the setting at 1,024 cells; otherwise the setting of 256 cells stands.
If the best of a grid lies on its edge in eta1 or eta2, one further value beyond the edge is run and this is said.

**A limit, stated now.** The settings are chosen by the loss on the same last tenth that the claim is scored on (with another seed). 48 settings an arm are compared; my reading is that the bias from this is small beside the differences looked for, but it is not zero, and the references of `nano/ref` may have chosen differently.

## The claim runs

Reported seeds 73129, 93683, 116089 (the first three of the five; all five only if the cost allows, which I do not expect). For each seed and each size (256; 1,024), 8 readings: arm LS at its setting; arm P at its setting; and each arm "born at nothing" at its setting if the arm's own setting has rscale > 0. Reported: mean and each seed.

Beside them: unsmoothed tables of the last k bytes, k = 1 to 6, counted on the stream, the longest context seen deciding, their loss made in the same crude way as arm LS's (`loop.py tables`; computed before this plan, since they involve no learner: 2.510, 2.070, 1.796, 1.755, 1.927, 2.155 nats and 27.0, 38.1, 47.2, 52.5, 53.7, 53.9% right; always the most frequent byte 14.9%; the byte frequencies alone 3.347 nats). Smoothed tables and GPT references are another line's (`nano/ref`); they are quoted in the report if that branch exists then.

## Predictions (mine, before any run of the step)

| | Prediction | Chance |
|---|---|---|
| A1 | 1,024 cells: arm P's loss is below arm LS's crude loss on each of the three seeds | 0.90 |
| A2 | 1,024 cells: arm P names more bytes right than arm LS (mean of seeds) | 0.65 |
| A3 | arm P, 1,024 cells: mean loss at or below 1.80 nats (about the unsmoothed table of the last 3 bytes, 1.796) | 0.50 |
| A4 | arm P, 1,024 cells: mean loss at or below 1.755 nats (the best unsmoothed table) | 0.20 |
| A5 | arm P's chosen size of the return strengths at birth is 0 or 0.03 | 0.70 |
| A6 | arm P, 1,024 cells: "born at nothing" and the chosen birth differ by less than 0.02 nats (or are the same arm) | 0.70 |
| A7 | arm P: 1,024 cells are at least 0.05 nats below 256 cells | 0.75 |
| A8 | arm LS's chosen eta1 is 1 (the value found best on prose in the tenth plan of `loop_small_case`) | 0.50 |

## Cost (measured: 14 s a reading with 256 cells, 60 s with 1,024, one core, 4 cores present)

Stage 1: 96 runs x 0.5 min = 48 core-minutes. Stage 2: at most 12 runs x 2 min = 24. Stage 3: 6 runs x 4 min = 24. Claim runs: at most 12 runs x 2 min + 12 runs x 8 min = 120. About 3.6 core-hours, under one hour of wall time. No run is longer than about 9 minutes.

## Not run in this step

Other fading rates; other births of the strengths from lines; steps that change over time; a least-squares arm scored through a fitted soft maximum; more than one step of relay; cells that hold; 4,096 cells and more, more readings and other D (step L2); the reported seeds 406364 and 545353 unless the cost allows.
