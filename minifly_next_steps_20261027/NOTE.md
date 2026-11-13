NOTE (planning record; no experiment, no claim promotion). It responds to Codex's SHARED-POLICY V1 (`SP-COND-V1`, commit 2d6a8fff on `codex/minifly-consolidated-20261023`). The report and design were read, not rerun.

# Next steps after SHARED-POLICY V1 (27 October 2026)

## 1. What SP-COND-V1 settles
This was the shared-controller fit requested in `minifly_review_v171r1_20261027/NOTE.md` §3, item 1. Fresh controllers were fitted by exact gradients on V171's frozen native checkpoints.

**Results (complete own answers out of 80):**

| | NATIVE | SGD3 | ADAM3 / ADAM1 / ADAM3-R0 | ADAM3-CONST |
|---|---:|---:|---:|---:|
| World 711 | 4 | 4 | 4 | 0 |
| World 712 | 11 | 11 | 12 | 0 |

* ADAM3 reduces teacher-prefix loss by about 0.32 bits/target relative to SGD3. Many parameters end at their caps.

**Readings, within the scope of these two exposed fixtures:**
1. **Optimisation is not the main bottleneck.** A stronger optimiser moves the parameters and lowers the loss, but adds at most one answer.
2. **On this graph, longer credit and memory-to-workspace feedback add nothing.**
   * ADAM1, ADAM3 and the structural R0 lesion end with identical correct sets.
   * This closes, for this graph, the question V171 was designed to ask.
3. **A shared policy reaches almost none of the per-prefix oracle envelope** (47/80 and 57/80). The R1 envelope reflected the oracle's freedom, not a reachable policy.
4. **Native content is used.**
   * Without it (CONST), the fit collapses to an immediate end-of-line output on all 80 items, although every class is expressible.
   * With real content, the end-of-line target is predicted correctly on 79 of 80 teacher prefixes. Failures are in the payload bytes, not in termination.

**Unresolved:** the parameter caps are active, so capacity remains confounded with fitting.

## 2. Recommendations
1. **Close the output-gate reader line; do not run the optional 2×2 instrument** (full-record credit × no projection).
   * Even its best case is bounded by a per-prefix envelope of about 60% of taught items. It sits on a native parent that fails E1, and neither of its arms is an admissible learner.
   * Neither outcome would change the next decision.
   * Worlds 711 and 712 have now been used by three successive rounds (V171, R1, SP). Any further qualification needs fresh worlds.
2. **Make native E1 the immediate target.**
   * The native parent recalls 3–6 of 64 taught OLD items and 1–5 of 16 NEW items after 16 passes. The CLS-round configuration (FE-PERM, homeostasis, adaptation reset) recalled 84–88% of taught facts.
   * The first question is why this lineage stores or expresses so little. In order:
     * **(a) Read-only diagnosis on the existing frozen checkpoints:**
       * per-item target margins in the native value;
       * KC overlap among the 80 prompts, since collisions were the CLS round's first finding;
       * whether the KC codes separate the items under an external evaluator readout, labelled as an instrument.
     * **(b) A small factorial of repairs already identified,** with no new architecture, scored on taught complete recall and retention on fresh worlds:
       * the P81 front end (fade 0.807 with permutation);
       * one-sided homeostasis;
       * adaptation reset;
       * the class-bias output port (V153: 2.13 of 2.38 bits recoverable);
       * even fan-out (V159).
3. **Owner decision before any re-entry work:** the operational no-attention boundary for state-dependent reads of native memory.
4. **Then the generic learner:**
   * next-byte prediction before every byte, with one circuit-local rule;
   * a qualified native substrate and fresh histories;
   * payload loss reported separately from end-of-line and syntax loss (CONST shows how a frequent endpoint can dominate);
   * surprise-gated writing tested against a relevance control.
5. **Registry:** cite Codex work by its identity (SP-COND-V1, V171-R1). Claude's next number remains V172.

None of these is launched by this note. Items 2(a) and 2(b) need the owner's go-ahead and caps.
