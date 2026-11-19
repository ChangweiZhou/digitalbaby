# Step-2 mechanics in the quick program (seed 73129, 16,384 bytes)

**Source.** This folder copies `reviewer_tissue_20261119` from the reviewer's branch at 0b0b6f8e. The copy is changed in `tissue.c` and `drive.py` only. The rest is unchanged: `README.md`, `deps/`, the reviewer's `out/*.json`, `batch*.txt` and `cmp_fixed.py`.

**The task.** It is the reviewer's task of 19 November 2026, 05:46 UTC. The governing texts are `THE_TISSUE_DEFINED_REVISION_5_20261119.md` and `TASK_STEP_2_THE_TWO_ENDS_CORRECT_THEMSELVES_20261119.md` (revision 1, unchecked).

## What was added

The option `told=1` of `drive.py` turns on the step-2 mechanics:
1. **The 256 output lines.** Line v carries the output of its one driver cell, `driver[v]` of the step-1 birth. These are the definition's terms, sentence 30 and C54.
2. **The emission.** At the reading step (φ = 31), the byte of the most active driver is emitted if its output exceeds 1/2, with ties going to the lowest byte value (sentence 75). Otherwise no byte is emitted. Emissions are computed in every run and saved in `out/<label>_emissions.npy`.
3. **The comparison.** At φ = 0 of position t, the world compares byte t with the emission of position t − 1. If they are equal, the yes line is set to 1 for that one step. Otherwise the no line is set, and that includes positions with no emission (sentences 77, 86 and 120, operation 1). Nothing is set at position 0.
4. **The 64 feedback cells.** The 32 yes-cells (FB[:32], inhibitory) receive the yes line, and the 32 no-cells (excitatory) receive the no line, each at strength 1. Each computes formulas 1 to 6 and 10, with θ = 0. Formulas 7, 9, 11 and 12 and the gain do not apply to them, and they get the same output floor as other cells.
5. **The count of right bytes.** The attempt for position t is the emission at t − 1, and no emission counts as wrong. The count is taken in windows A1 (bytes 2,048–6,143) and A2 (bytes 12,288–16,383).

With `told=0`, the feedback cells are skipped exactly as in 0b0b6f8e. Run silent, `coincidence gain=1 astar=1/32 floor=1/128` reproduces every field of the reviewer's `out/coinc_gain_a32_F.json` exactly: connections, makings, givings up, activity, gain, θ, ā, and the shares and scores in A1 and A2. Only the time differs.

The places where the brief and the definition differ or leave a choice were sent to the reviewer before the runs. They are restated in the commit message.

## Results

The command is `python3 -I drive.py <label> rule=<fixed|coincidence> told=<0|1> gain=1 astar=1/32 floor=1/128`.

In every run:
- the most frequent byte of the 16,384 is the end mark (byte 10);
- chance is 1/46 = 0.0217;
- always emitting the end mark scores 0.0630 in A1 and 0.0625 in A2.

### Share of bytes right, and emission

| Tissue | Feedback cells | Share right A1 | Share right A2 | Right bytes in A2 | Share of positions with a byte emitted | Yes / no given | Commonest bytes emitted |
|---|---|---|---|---|---|---|---|
| fixed | silent | 0.0000 | 0.0000 | 0 | 0.041 | 0 / 0 | 0, 4, 246 |
| fixed | driven | 0.0000 | 0.0000 | 0 | 0.060 | 0 / 16,383 | 0, 4, 5 |
| coincidence | silent | 0.0017 | 0.0002 | 1 | 0.367 | 0 / 0 | 0, 1, 4 |
| coincidence | driven | 0.0010 | 0.0010 | 4 | 0.315 | 13 / 16,370 | 0, 4, 1 |

### Cost and the state of the tissue

| Tissue | Feedback cells | ms a byte | Cells sending a step (mean) | Connections a cell (end) | S(1,0) A2 | S(2,0) A2 | S(4,0) A2 | Share neither silent nor constant A2, d = 1 / 2 / 4 |
|---|---|---|---|---|---|---|---|---|
| fixed | silent | 0.94 | 103 | 7.75 | 0.741 | 0.134 | 0.036 | 1.000 / 1.000 / 0.991 |
| fixed | driven | 1.15 | 198 | 7.75 | 0.759 | 0.153 | 0.029 | 1.000 / 1.000 / 0.991 |
| coincidence | silent | 11.10 | 102 | 19.43 | 0.764 | 0.128 | 0.041 | 1.000 / 1.000 / 1.000 |
| coincidence | driven | 9.16 | 131 | 17.18 | 0.729 | 0.141 | 0.041 | 0.941 / 0.960 / 1.000 |

## What the runs measured

- **No tissue emits the right byte above chance.** The emitted bytes are mostly values that never occur in the stream (0, 1, 4, 5, 246). Almost every position therefore receives "no".
- **Driving the feedback cells changes the tissue measurably:**
  - In the fixed tissue, cells sending a step rise from 103 to 198. Its connections cannot change.
  - In the coincidence tissue, cells sending a step rise from 102 to 131, and connections a cell fall from 19.4 to 17.2.
  - In the coincidence tissue, the share of cells neither silent nor constant in A2 falls to 0.94 at d = 1 and 0.96 at d = 2.
- **The scores S(d, 0) in A2** differ between silent and driven by 0.000 to 0.035. Their standard errors are not computed here, so these differences are not called changes.
- No learning result was expected, and none is claimed.
