# MiniFly V9.9B — a second relation family, with a clear limit

**Outcome:** The frozen REF learner used prior teaching to improve choices on eight never-reinforced comparisons in a randomized graded-order task. The prespecified final `W_L−N_L` choice difference across **64 untouched worlds** was **+9.57 percentage points** (paired 95% Student-*t* interval **+5.09 to +14.05 pp**). Absolute `W_L` choice was **59.38%** [54.89%, 63.86%], above 50% chance. The primary E3 gate and above-chance guard pass. The stronger, separately declared *graded-beyond-one-bit* gate does **not** pass: an oracle one-bit-per-symbol comparator can reach 75% on these withheld pairs. V9.9B therefore confirms a modest persistent teaching effect in a second synthetic relation family, but does not prove the learner acquired a full multi-level ordering rule. We do **not** promote a general persistent core from this result.

## Exposure, controls, and claim

Eight random uppercase bytes represented hidden ranks 1–8 in each world. Of 28 unordered comparisons, 20 were taught in both orientations, four presentations per orientation (**160 old records**). Eight fixed rank pairs were never taught in either orientation. Each symbol appeared in five taught pairs, equally often in both byte positions; outcome labels were balanced globally. The first held-out probes were read-only and preceded any feedback on those pairs. Three disjoint symbols formed the later 72-record cohort. The byte frame, REF value learner, V9.2 checkpoint, timing, and native readout were inherited unchanged from V9.9A.

`W` and `N` had the same initial state, bytes, labels, clocks, and engineered predictor learning; only `W` permitted the *old* native value writes. At the old-day prestate, both were forked into `R` (time), `S` (new sensory stream with native writes off), and `L` (the identical new stream with writes on). Final branches had the same clock. The scored answer came from the native value circuit, not an analytic decoder. An exact raw or canonicalized taught-pair table can only tie on withheld pairs; prior-feedback win counts per symbol can solve all eight. Exhausting all 256 binary-category assignments gives 75% as the best one-bit category-comparison score. This generous countermodel explains why a positive teaching effect alone is insufficient to claim a graded rule. The task tests learned feature reuse, not explicit transitive reasoning.

## Frozen pilot and independent confirmation

The [design](DESIGN.md) and [runner](run_v99b.py) were source-locked before the pilot. Engineering world 9902 passed read-only, equal-state, predictor, clock, write-ledger, code-separation, and analytic task checks. Fixture validation found no old/new/held-out cue overlap in all 77 planned worlds. The 12-world pilot (1401–1412) met its predeclared feasibility gate: old-end taught `W` choice 76.25%; held-out `W` 54.17% versus `N` 51.04% (only **+3.13 pp**); and both new-cohort `L−S` choice gains +36.11 pp. Pilot data were not included in confirmation and did not change the model, task, dose, or primary endpoint.

The confirmation worlds were 1501–1564. Choice is the fraction of correctly signed pair margins, with ties worth 0.5. Only final `W_L−N_L` was the confirmatory comparison; other intervals are descriptive. All values below are world means.

| Old held-out comparison probe | W choice | N choice | Paired W−N [95% interval] |
|---|---:|---:|---:|
| Old teaching end | 61.72% | 50.20% | +11.52 pp [+5.42, +17.63] |
| First day, before new cohort | 59.38% | 49.80% | +9.57 pp [+4.96, +14.18] |
| New cohort end, L | 57.81% | 47.85% | +9.96 pp [+6.43, +13.49] |
| Final day, R | 58.98% | 49.80% | +9.18 pp [+4.55, +13.81] |
| Final day, S | 58.98% | 49.80% | +9.18 pp [+4.55, +13.81] |
| **Final day, L — primary** | **59.38%** | **49.80%** | **+9.57 pp [+5.09, +14.05]** |

At the primary endpoint, 39 worlds favored `W`, 11 tied, and 14 favored `N`. The signed margin difference was +0.01814 [0.01290, 0.02338]. On *taught* old comparisons, `W` scored 74.53% at old end and 70.16% at final, versus `N` at 50.08% and 50.78%. This is genuine acquisition but imperfect even on trained pairs; weak generalization was already visible at old end, before later learning.

The later task was actually learned: new-pair choice at cohort end was **85.94% `W_L` / 87.50% `N_L`**, compared with **52.60% `W_S` / 49.48% `N_S`**. At the final day, `W_L` and `N_L` still scored 81.25% and 71.88%. Yet the old held-out `W−N` advantage changed by only **+0.39 pp** for `L−S` [−4.33, +5.11]; `S−R` was exactly zero in choice. The signed-margin `L−S` contrast was +0.00012 [−0.00080, +0.00105]. These results do not identify substantial write-induced loss under this disjoint-cohort schedule. They also do not imply that all future tasks are noninterfering.

## What the positive result does not establish

The one-bit comparator's 75% is an *oracle upper bound*, not its expected fitted performance; scoring below it neither proves nor disproves that REF uses only one bit. It means this run cannot exclude that coarse explanation. A [postrun per-edge diagnostic](results/HELDOUT_EDGE_DIAGNOSTIC.csv) shows uneven transfer: the six rank-gap-2 held-out pairs average `W−N` +6.77 pp, while two wide-gap pairs average +17.97 pp; one gap-2 pair is negative. Those comparisons are descriptive and were not the primary gate. The model has evidence of a useful old-teaching signal on the second generator, but not of reliable detailed rank recovery. As with V9.9A, these are synthetic byte relations, not a natural-language or broad task result.

The most direct bottleneck is now **quality of immediate relation extraction/readout**, not demonstrated erosion by the later disjoint learning episode: final held-out `W` was 59.38%, already only 61.72% immediately, and later new learning left the old causal advantage approximately unchanged. A subsequent bounded diagnostic could distinguish insufficient old-teaching dose from representation/readout limits, while keeping the same never-reinforced behavioral gate. A shared/private hybrid, eligibility rule, topology change, or evolution search should await such localization.

## Audit and compact bundle

The 64-world confirmation took **376.8 seconds**. The [independent reducer](audit_results.py) validated the 336 pilot and 1,792 confirmation rows, recomputed every published cell and interval from pair margins, checked the pilot gate, all source hashes, and the final acceptance record. There were no zero-distance or identical-support opposite-orientation code rows. The [metric tables](results/CONFIRM_METRICS.csv), [summary](results/CONFIRM_SUMMARY.json), [audit](results/AUDIT.json), and [manifest](MANIFEST.json) remain in this compact result folder; unchanged parent datasets and checkpoint are referenced by hash, never copied. Pilot worlds, confirmation worlds, and earlier V9.8/V9.9A worlds do not overlap.
