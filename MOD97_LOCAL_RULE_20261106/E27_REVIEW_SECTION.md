**E27 (Opus worker, 13 Nov 2026; branch `claude/sleepy-knuth-q96dd1` from `02897c6e`; E2 diagnostic, nothing trained).** The coordinator's `e27_measure.py` was fixed before the run. Its per-tensor output dropped the GRU's recurrent matrices and every bias. Its taught panel came from 6 of the 12 pairs. It loaded files without checking their hashes. The fixed script checks all 24 present files against their receipts, and every one matches. The split was confirmed by independent code to within 1.3e-7, and a version with the shares swapped fails, so the check can fail. The controls of revision 2 behave as required. In the positive control the two-bank amplification is 1.55 to 1.72 on all parameters and 2.8 to 3.3 below the top layer. The top layer is exactly 1, and the null's mean is 0.96 to 1.00. The scorer was hashed before the run.

Verdicts:
- **P27.1 misses** (0.53 / 1.41 / 0.80).
- **P27.2 is void** in all three seeds: usable under the exact path is negative.
- **P27.3 misses**.
- **P27.4 misses** and **P27.5 misses**, scored on all four life arms after the six E26 life weight files arrived (`ce409d64`; hashes match the receipts). Part A is bit-identical on the rerun.

The supplements disagree with the registered figures, so the result is mixed.
- On the sweep-born answer-target file, the sign path amplifies the **distractor letters'** error below the top layer by 1.95 / 2.38 / 2.77, outside the null in all seeds. It damps the structural bytes' error to 0.2 to 0.4, and the two effects cancel in "other". S1 is therefore 0.63 / 1.59 / 0.96.
- S3 (backprop-born, every byte, below the top layer) is 2.18 / 1.05 / 3.13.
- Every trained GRU file is amplified about 1.7 to 2.9 below the top layer.
- On all answer-trained files, the *exact* path's flood is already 89 to 605 times the answer's. The answer direction is swamped under both paths, which makes "usable" uninformative.
- S6 with all four arms: the sweep-lived untaught-minus-taught gap is larger than the backprop-lived one in 2 of 6 within-birth comparisons and in 0 of 3 seeds by arm means, with every contrast between −0.08 and +0.05. S6 does not support the composition reading.
- Amplification below the top layer does not mark the collapse. The collapsed birth (1.25 / 1.42 / 3.05) sits inside the range of births that did not collapse, and the balanced birth, which did not collapse, is highest (12 to 17). Unlike the coordinator's reading, though, not every trained network is amplified on this aggregate: the sweep-born answer-target file is amplified in 1 of 3 seeds (in all three on letters alone).
- E27's sign path is exactly the gradient E24's sweep births computed: relative difference 0 or 4.6e-7 under `birthrule T, traceup all`. The births stepped it through AdamW, and the balanced birth weighted its loss ½ answer + ½ other rather than uniformly.

Nothing here says what happened at the weights where E24 collapsed, because those weights were never saved.
