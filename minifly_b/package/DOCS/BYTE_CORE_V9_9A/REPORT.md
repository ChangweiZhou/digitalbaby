# MiniFly V9.9A — REF E3 confirmation

**Result:** The frozen REF learner retained a positive choice advantage on never-reinforced symbol combinations after the original delay, a separate 72-record cohort, and a further day. Across 64 new randomized worlds, the prespecified final held-out `W_L−N_L` choice difference was **+17.19 percentage points** (paired two-sided 95% Student-*t* interval **+10.39 to +23.98 pp**). This passes the sole locked confirmation gate. It confirms **limited E3 reuse within one synthetic relation family**; it does not establish a general persistent learning core or NLP competence.

## What was actually learned and scored

The unchanged [V9.8 E3 generator](../BYTE_CORE_V9_8_TRIAL/DESIGN.md) randomly divides six uppercase-byte symbols into three P and three N symbols in each world. It teaches both orientations of six of nine P–N pairings, then asks for the aversive-versus-neutral choice on the other three pairings. Each scored combination is absent from all feedback records, and the first read-only probe precedes any feedback on it. Each symbol occurs in two taught edges and equally often in both byte positions. A strict raw or canonicalized **pair-key** lookup has no taught answer for the held-out pairs; position or symbol-frequency shortcuts have no fixed label; a one-bit class rule inferred from teaching can solve all three. That simple learned rule is a legitimate but narrow E3 explanation. The later new cohort uses three disjoint symbols.

`W` and `N` start from identical frozen V9.2 parent state and see the same old bytes, outcome bytes, labels, and timing. Only `W` permits the old native value writes. Both update the engineered byte predictor identically; the **native value circuit**, probed on a read-only clone, supplies the scored value. At the old-day checkpoint, each state forks to `R` (time only), `S` (identical new sensory stream and teaching calls with new native writes disabled), and `L` (same stream with new native writes enabled). Every final branch is scored at the same clock time. `W−N` therefore tests whether prior native teaching improved novel-pair choices; `S−R` and `L−S` diagnose what the later exposure and write flags add under this schedule. The analytic class decoder is a task-solvability witness, not part of the fly.

## Results

Choice is the mean of three held-out pair comparisons per world; ties score 0.5. Values below average 64 world-level scores. Only the **final L** interval is confirmatory; all other intervals are descriptive, without multiplicity adjustment.

| Held-out probe | W choice | N choice | Paired W−N [95% interval] |
|---|---:|---:|---:|
| Old teaching end | 73.44% | 49.48% | +23.96 pp [+15.38, +32.54] |
| After first day, before new cohort | 67.71% | 48.96% | +18.75 pp [+11.81, +25.69] |
| New cohort end, L | 64.58% | 46.88% | +17.71 pp [+10.15, +25.27] |
| Final day, R | 67.19% | 48.96% | +18.23 pp [+11.28, +25.18] |
| Final day, S | 67.19% | 48.96% | +18.23 pp [+11.28, +25.18] |
| **Final day, L — primary** | **66.67%** | **49.48%** | **+17.19 pp [+10.39, +23.98]** |

At the primary endpoint, 31 worlds had positive `W−N` choice, 28 tied, and 5 were negative. The signed value-margin difference was +0.04906 [0.03828, 0.05985], positive in 55/64 worlds. This analog signal is distinct from thresholded choice. The final trained-pair guard was 85.42% for `W_L` versus 50.52% for `N_L`; held-out choice was substantially lower than trained-pair choice.

At the final probe, `Δ_S−Δ_R` was exactly **0 choice points** in all worlds; the maximum difference across all corresponding R/S pair margins was 3.6×10⁻¹⁵, numerical precision. Thus the extra sensory stream without new native writes left the **final measured values** unchanged in this assay. `Δ_L−Δ_S` was **−1.04 pp** [−5.96, +3.87], so a new-write effect on the *old-teaching advantage in choice* is unresolved. Its signed-margin contrast was −0.00213 [−0.00383, −0.00043], a small descriptive decline. Neither contrast is a universal claim about forgetting: the sample contains one new-cohort schedule and one synthetic relation, and the choice interval permits modest help or harm.

**Postrun check that the new cohort was actually learned.** The locked primary did not score the three new-cohort pairs, so a separate [read-only diagnostic replay](audit_new_cohort.py) added those probes without changing any trial endpoint. It reproduced all 1,792 original pair-margin vectors exactly and scalar metrics to 10⁻¹². At new-cohort end, new-pair choice was **85.94% W_L and 89.06% N_L**, versus **48.44% W_S and 45.31% N_S**. One day later it was **73.96% W_L and 70.31% N_L**, versus **45.83% W_S and 44.27% N_S**. Thus later native writes did produce substantial, partly durable new learning in both old-teaching conditions. The probe is postrun and diagnostic, not a second confirmation test; see [NEW_COHORT_DIAGNOSTIC.json](results/NEW_COHORT_DIAGNOSTIC.json).

## Audit, scope, and next decision

The locked design fixed the 64 science worlds (1301–1364), REF architecture, V9.8 generator, primary endpoint, interval rule, and branches before any science-world outcomes were inspected. The 1201 engineering preflight reproduced all 16 archived V9.8 REF E3 probe rows exactly and checked shared prestates, clocks, predictor equality, inert no-write labels, and unseen held-out contexts. The science run completed all 64 worlds in 325 seconds. An [independent reducer](audit_results.py) checked all 1,792 rows, their pair-margin-to-choice arithmetic, unique roster, finite values, all source hashes, and the published summary; it found zero opponent-code collisions. See the compact [metric table](results/WORLD_METRICS.csv), [summary](results/SUMMARY.json), [audit](results/AUDIT.json), and [source/asset manifest](MANIFEST.json). The bundle is under 1 MB and copies no source data or checkpoint.

The result changes the diagnosis: REF has a reproducible, delayed **within-family** held-out choice signal that cannot be credited to direct reinforcement of the tested combinations. It remains a modest 66.67% absolute choice on three pairs per world, and a learned one-bit class per symbol suffices. The next promotion test should prospectively specify a materially different relation or downstream task, retain the paired no-prior-write control, and check new-task competence. A larger bank, topology search, or evolution run is not yet justified by this result alone.
