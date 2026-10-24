# Step 6 result: stable context features restore E1 writing, but the full gate stays open

**Fixed decision: `H4_E1_COMPONENT_NOT_DEMONSTRATED` under the prespecified
three-part gate.** All 16 new worlds (9601–9616) completed, the technical
world 9599 passed, and the independent receipt audit passed. The new H4
projection does show a clear native Full151 write effect on these *taught*
ordered cues: bout-12 choice was **100.0%** with writing versus **54.2%**
without, a paired **+45.8 percentage-point** effect (two-sided 95% Student-t
interval **+27.6 to +64.0 pp**). The protected FE0 control showed **90.6%**
versus **51.0%**, paired **+39.6 pp** (interval **+22.2 to +57.0 pp**).

The old next-byte-probability Q projection, run on the *same fresh worlds*,
scored **68.8%** with writing versus **54.2%** without. Its paired effect was
**+14.6 pp** (interval **−5.9 to +35.1 pp**). The prespecified difference
between H4's and Q's write effects was **+31.3 pp**, but its interval was
**−2.1 to +64.6 pp**. Because the lower bound is not above zero, the full
promotion gate **fails**. H4 has its own positive controlled E1 result; the
data do not meet the frozen standard for superiority to Q. A post-hoc sign
check was also inconclusive: 10 positive among 14 nonzero world contrasts
(two-sided p≈0.18).

| Route | Pre: reversed-pair KC distance | Bout 6 | Bout 12 | Bout-12 W−N choice |
|---|---:|---:|---:|---:|
| FE0 | 7.5 | 8.3 | 8.3 | +39.6 pp |
| Q probability projection | 63.2 | 36.6 | 9.2 | +14.6 pp |
| H4 fixed context features | 151.8 | 149.9 | 149.9 | +45.8 pp |

The Q route's mean pairwise next-byte-distribution L1 distance fell from
**0.818** before teaching to **0.552** at bout 6 and **0.129** at bout 12.
Its KC distinction fell with it. H4 uses the unchanged V9.2 four-byte
feature *identities*, rather than those changing next-byte probabilities,
and retained its KC distinction. Every route had **184 active KCs** in the
pre-outcome eligibility state throughout training. Thus missing eligibility
or shrinking KC occupancy does not explain Q's loss. These observations
support the predicted probability-convergence mechanism, though they do not
prove it is the only cause of the value difference.

All 16 H4 written-world scores were 1.0, so their Student-t absolute-score
interval in the machine summary collapses to [1,1] and should not be treated
as a calibrated uncertainty interval. As a robustness description, the
two-sided 95% exact binomial lower bound for 16/16 world-level perfect scores
is **0.794**, still above 0.5. This additional check does not alter the
failed H4-versus-Q gate.

**Evidence boundary:** This is E1 acquisition on six explicitly reinforced
ordered cues, 12 exposures per cue. A frozen exact-key lookup table could
also pass. No unseen byte presentation, never-reinforced relation, delayed
interference, fusion, autonomous choice, byte emission, or arithmetic was
tested. The result establishes that one label-free fixed feature projection
can let the existing Full151 value circuit learn these bytes without changing
its synaptic rule. It does not establish the project's persistent learning
core or justify replacing FE0 in a general system yet.

The technical receipt checks FE0 standalone state equality after every
record, equal predictor state across routes, read-only probes, deterministic
feature identity, clocks, write eligibility, and teaching counts. The audit
checks source hashes, fixed world roster and schedule digests, row arithmetic,
feature constancy, and intervals without replaying native dynamics. Its
in-memory arithmetic-tamper check rejected an altered choice score. No
completed world was rerun or replaced.

Before another science run, an **E0 code census** should test whether
prespecified label-preserving presentation changes retain usable H4 codes.
The exact four-byte hash may simply rekey them. Only if that access check is
credible should a fresh **E2 changed-presentation test** compare H4 with FE0
on never-reinforced forms, scoring the first choice before feedback. The
present exact-cue success does not license another memory topology search or
a claim of unseen-relation transfer.

Artifacts: [design](DESIGN.md), [runner](step6_context.py),
[audit](audit.py), [summary](results/SUMMARY.json),
[source lock](results/SOURCE_LOCK.json), and
[technical receipt](results/TECHNICAL.json).
