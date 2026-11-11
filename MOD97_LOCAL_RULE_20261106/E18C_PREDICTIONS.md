# E18c predictions (committed before the first R2 birth, 11 November 2026)

Plan `E18B_E18C_E23_PLAN_20261111.md` (v3), step 2. B2 = `E16_HEADS=local` (relay cut), R2 = `E16_HEADS=relay` (only the first bank's state → second bank's projection opened; the main readout's direct reads of the first bank stay detached), both `E16_BIRTHRULE=T E16_TRACEUP=all`, same heads, losses, parameters (1,782,418), seeds 73129 / 93683 / 116089, same host. `e23_check.py`: with `lp2`'s weights zeroed the two give identical gradients. Lives: one `T` life from each birth under the birth's own heads mode (package comparison), `E16_REPLAY=raw E16_CAP=65536`.

Hard mean = (add89 + sub89)/2 at k = 12, main head, whole answers, final checkpoint; checkpoint mean = its mean over the six checkpoints; F1 = mean of the four mod-97 operations at k = 12 in the life.

- **PC.1** R2 hard mean ≥ B2's + 5 in ≥ 2 seeds. *~50%.*
- **PC.2** R2 checkpoint-mean hard mean ≥ B2's + 5 in ≥ 2 seeds. *~55%.*
- **PC.3** R2 birth CPU-s within 15% of B2's in every seed. *~80%.*
- **PC.4** (package) After block 6 of the life, R2-life F1 ≥ B2-life F1 + 3 in ≥ 2 seeds. *~50%.*
- **PC.5** R2's first-bank-head hard mean at birth ≥ B2's + 5 in ≥ 2 seeds. *~50%.*

Reading: PC.1/PC.2 hold → error passed from the second bank to the first is worth points with everything else fixed; both miss → E18's package gap did not come from error passed between banks (the main head's direct reads, `aux` minus `relay`, are untested: NOT_RUN). PC.4 is a statement about birth + life rule together. Crossed lives NOT_RUN.
