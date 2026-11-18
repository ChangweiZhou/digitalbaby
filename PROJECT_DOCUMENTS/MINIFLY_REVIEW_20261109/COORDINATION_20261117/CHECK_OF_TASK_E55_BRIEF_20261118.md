# Check of the brief of E55, first part (revision 1), before its release

18 November 2026. Written for the outside reviewer by an agent that was given the brief and the files it builds on, and not the author's reasoning. The check is a reading and a recomputation with numpy; no code of another branch was run.

**Verdict: (in preparation; this file is written while the check proceeds, and the verdict is entered when the check is complete). The notes below are working notes and are replaced by the final text.**

## Working notes (findings so far, 18 November)

1. Record, wrong: "In the arm `open` the gates did not open. The new task was learned there ... in the old cells". From the weights of row 4,096 of `open` on `unrelated` (script part `open`): 62 (sweep) and 53 (back-propagation) of the 128 cells in reserve have a mean over items of 1 − z above 0.5 at some position (57 and 38 of them at the position of the first symbol), although only 1 and 8 have a mean over positions and items above 0.1; with the cells in reserve held shut the never-taught score is 0.7 and 5.7 (receipts) against 100 and 100. On `conflict` the sentence holds (no cell above 0.5 at any position; 83.0 and 87.6 with the cells shut against 85.0 and 94.9).
2. Measurement: "cells in reserve open by more than 0.1" is a mean over positions and items; a cell that opens at one position of 19 or 22 has a mean of at most 0.053 or 0.045 and is not counted. The licensed sentence "learning did not open the cells in reserve" and P55.6 rest on this count.
3. Check 1 limits can be met (script part `start`): largest 1 − z 4.541e-5; largest state 5.6e-4 and 5.9e-4; largest added unit 2.9e-4 and 3.6e-4; largest difference of an output 2.4e-3 and 2.5e-3; answers that differ 1 and 0 of 32,884; block residues 1.9e-7 and 1.3e-7 at the input map.
4. Under the sweep the entries that the projection brings to zero are rounding remainders of size 1e-16 with either sign (recurrent matrix, gate rows of the reserve, columns of old cells: 15,913 of 16,384 non-zero; `lm`, rows 256 to 511, columns of old cells: 31,813 of 32,768 non-zero); the sweep sends the error back by the sign of each weight.
5. At the start `shut` the gates are shut on the new tasks' items as well (1 − z at most 5.7e-5). Moving a gate's summed input by 8 on one byte of a new task needs at least 245 to 1,507 updates with steps of at most 0.001 a weight.

## Defects

(in preparation)

## What was verified and found sound

(in preparation)

## What could not be checked

(in preparation)
