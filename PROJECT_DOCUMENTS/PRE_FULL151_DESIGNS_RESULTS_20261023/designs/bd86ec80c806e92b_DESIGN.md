# FULL_151 byte-bridge behavioral comparison, v1

Frozen before science-world outcomes. This is the next narrow experiment after the E0 byte-to-FULL_151 bridge and W4's unsuccessful arithmetic development pilot. It asks whether the **already fixed** V9.2 predictive-PN bridge improves the inherited FULL_151 memory's choice on never-reinforced byte combinations. It does not test four-operation arithmetic, a new memory mechanism, or free-running byte generation.

## Claim and level

Use the V9.8/V9.9A E3 synthetic two-class relation. Six random uppercase symbols form three aversive and three neutral symbols. Both orientations of six cross-class pairs receive feedback twelve times; three pairings (both orientations) never receive feedback. Each symbol occurs in two taught pairings and equally often in both positions. The held-out prompt presents the two orientations in a randomized left/right order. A fixed policy reads FULL_151's association value for both options from the same pre-choice state, chooses the **higher-value (neutral)** option, and emits `L` or `R` **before any held-out feedback**. This is a prompted two-choice action, not spontaneous text production. Its accuracy is exactly the native value-sign choice used in the earlier E3 assay, so the comparison preserves a proven task/readout. The action policy has no learned weights and sees no answer label.

The four prospective arms share one parent V9.2 checkpoint, world, byte stream, teaching bits, predictor learning, and clocks: `BRIDGE_W`, `BRIDGE_N`, `FE0_W`, `FE0_N`. `BRIDGE` uses the previously fixed gain 0.5 predictive-PN projection; `FE0` sets gain 0. In `W`, native FULL_151 writes are enabled during the old 144 records; in `N`, only those native writes are disabled. All four receive the 72-record disjoint new cohort with native writes enabled, then one final day of passive time. The predictor learns identical bytes in all arms. No mode, gain, dose, relation generator, or scoring threshold is tuned in this study.

The sole primary is the 64-world paired mean difference-in-differences in final held-out choice:

`(BRIDGE_W − BRIDGE_N) − (FE0_W − FE0_N)`.

Worlds 2001–2064 are the independent analysis units. A two-sided 95% Student-t interval is reported. A lower bound above zero supports incremental bridge benefit in this family; an upper bound below zero supports harm; otherwise the effect is unresolved. Independently report both absolute W choices, each W−N contrast, immediate taught-pair choice, signed value margins and world signs. A positive analog margin cannot replace the choice primary. No optional stopping or world substitution.

## Exposure, countermodels, controls

The score uses only the three held-out pairs, absent from both old and new teaching. A raw or canonicalized exact-pair lookup is at chance on them. A learned one-bit class per symbol can solve the task, so any success remains narrow E3 within this synthetic relation. Position and symbol frequency are balanced; pre-exposure shortcut performance is measured by each N arm. The analytic class oracle is 100% and is external to the fly. A valid E3 attribution requires a positive old-write contrast, not merely high BRIDGE_W choice. The paired FE0 arm tests whether the bridge adds to an already functioning learner. A context-lesioned bridge probe at final time is descriptive and cannot substitute for the prospective causal comparison because its native memory was trained under the intact bridge.

The two options are supplied by the environment. `L`/`R` output uses a fixed value comparator; this experiment does **not** establish a learned output head or autonomous detection of when to answer. W4's 513-way action-credit mechanism is excluded because its negative-credit direction failed in 5/32 development prefixes. W4's raw-answer benchmark is not part of this learner.

## Engineering gate, cost, and audit

Before science, run focused tests of subclass clone/restore and owner rebinding, bridge/FE0 state parity at birth, identical predictor updates, read-only choice purity, held-out exposure exclusion, fixed output ordering, and exact FE0 regression on archived V9.8 world 1201. The latter technical world is excluded from inference. If engineering fails, stop without science. Once technical checks pass, run all 64 fixed worlds, saving one compact atomic receipt per world and resuming only missing receipts after operational interruption. Do not rerun a completed world or restart after scientific failure. Expected cost is minutes, based on V9.9A's 64-world 325-second run, but four full arms may be slower; no hard runtime guarantee. Use `caffeinate` attached to the runner. Do not save full event rows or copy parent assets.

An independent reducer recomputes choices, paired effects, intervals, arm/world counts, source hashes, predictor equality, old/new native write ledgers, clock equality, and exposure disjointness from receipts and regenerated fixtures. The final report states the actual E0–E4 level and the limits above regardless of outcome.
