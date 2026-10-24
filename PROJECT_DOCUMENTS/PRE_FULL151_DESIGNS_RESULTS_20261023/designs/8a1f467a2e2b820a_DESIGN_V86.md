# V86 design: common-world bridge and uninterrupted-life stress test

## Question

V83 showed that a candidate can look attractive in branched challenges yet fail under accumulated life history. V84E/V85E subsequently improved damage/new-learning profiles under their own confirmation protocol. V86 tests the missing bridge: whether those gains persist when all five frozen candidates experience identical fresh raw-PN worlds and, separately, one uninterrupted state history.

## Frozen candidates

1. REF.
2. V82 champion.
3. `S82_D81`: V82 static genes plus V81 dynamic genes, with the exact V82 realized PN→KC matrix copied into the hybrid.
4. V84E champion from the uploaded Claude archive.
5. V85E champion from the uploaded Claude archive.

V84E/V85E use the same audited `model_evo.py` and `common_evo.py` hashes as V82E, so this is one engine, not a cross-implementation comparison.

## Attribution controls

Raw PN inputs are identical across arms. Each candidate may encode those inputs differently because PN→KC weights and KC sparsity are candidate genes. The primary evaluator uses the unmodified REF Q projection for every arm. Candidate-coupled reader and actual-output measures are shadows. Nonstructural hybrid attribution does not redraw the PN→KC matrix.

## Branched panel

This preserves the V83 structure at loads 48 and 96, with six foundation exposures per pair, 12 challenge bouts (six for NEW), STABLE/ANOMALY1/ANOMALY2/REV/NEW, a 12-pair mixed-learning suffix, 0/24-hour endpoints, and matched no-write challenge branches.

This panel is the bridge to the existing V83/V84E/V85E result families.

## Life panel

No episode resets the learner. After learning 48 identities, twelve existing memories are divided into four stable, four anomaly, and four reversal targets. After the challenge and a one-day delay, learning proceeds to 72, 96, and 120 identities. Up to four additional epochs add 12 identities each and challenge 12 older memories drawn only from identities 48 and above, reaching loads 132, 144, 156, and 168.

The 36 identities from the initial 48 that were not selected for the first challenge are never directly challenged later. They are the clean early-retention cohort.

## Long-life diagnostics

- **Acquisition curve:** new identities are probed after exposures 1, 2, 4, and 6 without altering state.
- **Revision curve:** reversal targets are probed after cycles 1, 2, 4, 8, and 12.
- **Untrained specificity:** pairs 168–175 are never learned; the evaluator measures strong unsupported responses in either direction without assigning them an arbitrary correctness label.
- **Age-local damage:** pairs 176–177 are learned only on copied states. A matched copy receives the same sensory/reinforcement schedule with plasticity disabled. Damage to all currently learned memories is compared immediately and after one day. These probes cannot contaminate the main life.
- **Margins:** actual, fixed-reader, and a joint threshold margin are retained, preventing accuracy alone from hiding erosion toward the decision boundary.

## Inference

Independent history/seed is the replication unit. Contrasts are paired against REF within the same raw world. The runner does not treat individual memories as independent replicates and does not define a one-number winner.

A useful result must be interpreted across acquisition, revision, untouched-memory retention, local damage, and specificity. A candidate that preserves old choices only by becoming hard to teach is not a persistent-learning solution; nor is a candidate whose better starting margins merely conceal unchanged damage.
