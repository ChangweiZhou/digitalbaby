# Step 13 — 16-item byte-memory load test

**Outcome: the prespecified uniform 16-item E1 gate did not pass.** With the unchanged Step-6 fading-pair input and four native Full151 output stores, the model correctly emitted the taught byte on **461/512 (90.04%)** final canonical probes across 32 new worlds, versus **128/512 (25.00%)** when native writing was disabled. That is a strong teaching-associated result at higher load. However, the frozen gate also required *each* of the 16 cues to be correct in at least 24/32 worlds; `1+1=` reached only **12/32**, `3+1=` **19/32**, and `1+3=` **23/32**. The run therefore remains `BOUNDED_E1_16_NOT_DEMONSTRATED`. FE0 also failed, with **268/512 (52.34%)** taught-cue accuracy.

This is a bounded **E1 diagnostic of byte-cue to taught-byte recall**. It does not demonstrate addition, new-sum generalization, a persistent learning core, or one architecture that passes both the Step-6 recitation and Step-11/12 held-out relation tasks. Its four complete Full151 output stores remain an engineering scaffold rather than a compact output organ.

## Frozen exposure and controls

The [design](DESIGN.md) fixed all 16 ordered `a+b=` byte cues with `a,b ∈ {0,1,2,3}`. Nine cue-to-answer assignments embed the old A/B/H crowding patterns; the other seven assignments are independently shuffled from a balanced four-byte answer multiset in each world. Each answer byte `0`–`3` occurs for exactly four cues per world. These are arbitrary taught labels, not arithmetic targets. Every canonical cue received 12 interleaved teaching exposures. An exact-key table can solve the primary E1 task; the experiment asks whether this particular Full151 interface can do so under the stated load and write history.

FE0 and the frozen FE0+fading-pair input saw the same cue bytes, answer bytes, 192-record order, and clocks within each world. W made four native output-store writes per record (**768** total); N received the same sensory and teacher events with native writing off. The learner emitted one output byte at `=` **before** the arriving answer byte taught its channel. Probe outputs were obtained on read-only clones without feedback. The `7 spaces || a || space || +b=` form was never taught and was scored only as a secondary E2 spacing diagnostic. A fixed whitespace canonicalizer plus exact-key memory could solve that form, so it is not an E3 test.

Technical world 13000 was excluded. Eight pilot worlds (13001–13008) met their frozen feasibility gate: pair W canonical accuracy **89.06%**, W−N **+64.06 percentage points**, and each answer class at least 50% correct. The 32 confirmation worlds (13101–13132) were new, disjoint from the pilot and all earlier science, with no replacements or reruns. Neither the code nor gates changed after source locking.

## Confirmation result

| Final probe, 32 paired worlds | FE0 | FE0 + fading pairs |
|---|---:|---:|
| Canonical, W correct | 268/512 (**52.34%**) | 461/512 (**90.04%**) |
| Canonical, N correct | 128/512 (25.00%) | 128/512 (25.00%) |
| Canonical W−N, world-paired 95% interval | +27.34 pp [+23.27, +31.42] | **+65.04 pp [+63.07, +67.01]** |
| Minimum per-cue W correct | 2/32 | **12/32** |
| Untaught spacing variant, W correct | 234/512 (45.70%) | 439/512 (85.74%) |
| Untaught spacing variant, N correct | 128/512 (25.00%) | 128/512 (25.00%) |

The descriptive difference between routes in canonical W−N was **+37.70 pp** for pair over FE0 (world-paired interval **+32.56 to +42.83 pp**). Pair passed the frozen overall 85% and positive W−N interval requirements; it failed the per-cue requirement. FE0 failed both overall and per-cue requirements. The preferred-sufficient-route field is consequently `NONE` under the protocol, even though pair is substantially better than FE0 on aggregate.

The three pair-route canonical per-cue misses are `1+1=` **12/32**, `3+1=` **19/32**, and `1+3=` **23/32**. For `1+1=`, whose taught output is byte `2`, the model emitted byte `3` in **20/32** worlds. This makes a concrete interference candidate, but does not identify its cause. The excluded technical world's pair-route KC supports were all distinct; the nearest support to `1+1=` differed in **96** KC bits. Thus these receipts do not support an *exact* KC collision explanation. They also do not distinguish shared-feature overlap, opponent-channel suppression, accumulated writes/time, or later Full151 readout dynamics.

The spacing E2 result is descriptive and uneven at this load: pair W was **85.74%** overall, while the variant of `1+1=` was correct in only **9/32** worlds. Step 6's four-fact E2 success therefore cannot be promoted to a reliable 16-item changed-presentation claim. Its original small-panel result remains valid within that scope.

## Integrity, interpretation, and stop

The independent [auditor](audit_step13.py) regenerated every world fixture without importing or running the learner, checked all 32 confirmation receipts, labels, streams, answer-derived teacher routes, native write counts, clock and sensory parity, read-only probe rosters, native four-value argmax outputs, and all reported reductions. The [technical audit](results/TECHNICAL_AUDIT.json) rejected six deliberate corruptions of a receipt. [Source lock](results/SOURCE_LOCK.json), [pilot audit](results/PILOT_AUDIT.json), [summary](results/SUMMARY.json), [final audit](results/AUDIT.json), and compact [world receipts](results/confirm) are retained. There was no execution failure. The 32 confirmation worlds (four branches per world) took a summed **714.6 seconds** of execution; pilot and technical worlds added **173.2** and **21.8** seconds, respectively. The folder occupies about **2.3 MB** and contains no copied checkpoint or raw event trajectory.

The result shows that the established byte front end plus Full151 can support substantial directly taught byte recall at 16-item load, but **not uniform reliability across the 16 tested cues** under the frozen rule. The earlier Step-12 lack of held-out E3 choice with the pair route is a separate limitation; this E1 improvement does not repair it. Relative to one Step-6 panel, increasing from four to 16 cues also quadrupled total records, writes, and simulated learner time at fixed per-cue exposure, so this run measures practical load tolerance rather than isolating an address-capacity mechanism.

No second mechanism experiment was started. If this line continues, the narrow next diagnostic is the already identified target-only versus current opponent teaching rule on these same 16 cues, with the same byte streams and explicit acknowledgement that total plastic writes change. The primary observation to explain is `1+1=`'s persistent wrong-byte-`3` output. That comparison should be frozen as a new experiment; the Step-13 threshold cannot be revised retrospectively.
