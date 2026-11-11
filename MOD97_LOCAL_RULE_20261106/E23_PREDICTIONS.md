# E23 predictions (committed before the first E23 birth, 11 November 2026)

Plan `E18B_E18C_E23_PLAN_20261111.md` (v3), step 3. All cells: `E16_HEADS=local E16_BIRTHRULE=T E16_TRACEUP=all E16_HEAD1LOSS=all` (first bank's head on every next byte; main head on answer bytes). H0: no extra head. N1 / F4 / F4s: a third head on the second bank's state only, trained on the byte 1 / 4 / 4-shuffled-within-item ahead, at identical input positions (those whose byte 4 ahead is a prompt byte), loss weight 1; 2,176,915 parameters. d = 4 fixed in advance. B2 = the same-host E18b partner (no `HEAD1LOSS`).

Hard mean = (add89 + sub89)/2 at k = 12, main head, whole answers, final checkpoint.

- **P23.1** F4 hard mean ≥ N1's + 5 in ≥ 2 seeds. *~20%.*
- **P23.2** F4 hard mean ≥ F4s's + 5 in ≥ 2 seeds. *~25%.* (F4s is not timing-free: it teaches the item's own prompt bytes at random positions.)
- **P23.3** N1 hard mean ≥ H0's − 5 in ≥ 2 seeds. *~50%.*
- **P23.4** H0 hard mean ≥ B2's − 5 in ≥ 2 seeds. *~50%.*
- **P23.5** (diagnostic) F4's far-head top-1 on prompt targets, pooled over the four k = 12 panels, exceeds the structural baseline (most frequent target given input-byte class × target class, on the same items) by ≥ 3 points in every seed. *~30%.*

Decision table: P23.1 and P23.2 hold → a later target with its timing is worth a life test and the credit-horizon crossing; P23.2 alone → timing matters but not more than a near target; both miss → no benefit from a later *prompt-byte* target in this package at this dose (answer-bearing later targets untested); P23.3 or P23.4 miss → extra every-byte heads under sign feedback damage the bank they train and the contrasts cannot be read as timescale tests. If F4 < N1, target entropy explains it as well as timescale. Births only; credit-horizon crossing, cross-item control, lives: NOT_RUN.
