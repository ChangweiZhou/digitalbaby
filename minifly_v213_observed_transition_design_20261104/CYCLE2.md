# Cycle 2 · Local observation update

Refine. Compute the gate and emission increments from one prediction snapshot. The reveal writes only the selected table rows and one opcode row. An earlier unscored event in a probe changes transient state but receives no update.

Audit. Compare the local increments to finite differences, check exact no-write behavior, and verify that the outcome arrives after prediction. Match the shuffled assignment counts within teaching blocks and steps.

Test. The analytic increments agreed with finite differences. All four arms consumed the same eight packets. Aligned and shuffled credit each wrote eight times to each opcode, and the no-write arm retained its birth parameters.
