# MiniFly V9.9C — more writing helps a little, but does not close the transfer gap

**Result.** The prospectively specified 64-world comparison found a small positive effect of the extra four early old-teaching bouts. At the eighth-bout, pre-feedback probe of **never-reinforced rank pairs**, `D−L` native choice was **+2.34 percentage points**, paired 95% Student-*t* interval **+0.13 to +4.56 pp**. This passes the prespecified zero-effect test, while the interval's upper bound is below the prospectively defined **+5 pp practically interesting gain**. Extra early writes therefore contribute a little, but doubling native value writes under this schedule is not a substantial **immediate** repair. The effect on held-out choice after a day and later learning is unresolved. An external, train-label-only linear probe decoded substantially more rank information from the same frozen KC code; that probe is a diagnostic, **not** the fly's behavior.

## What was tested

This experiment reuses the frozen V9.9B graded-rank task and REF value learner. Each new world randomly maps eight uppercase bytes to ranks. Twenty undirected comparisons, both orientations, receive feedback; eight other comparisons, both orientations, never receive feedback. Every branch gets the **same 320 old sensory records**, labels, sequence, predictor learning, and clock. Only native value-write flags differ: `Z` writes 0 times, `E` writes in the first four 40-record bouts (160 times), `L` writes in the last four (160), and `D` writes throughout (320). The first four bout permutations match the V9.9B generator. `D−L` isolates the effect of the extra early writes given identical later sensory history and later write flags; it is not a pure context-free plasticity-rate estimate.

The native value circuit makes all behavioral choices on read-only clones. No held-out comparison is reinforced before or after its first probe. An exact-pair lookup can only tie on held-out pairs (50%). A win-count rule **computed only from taught labels** solves them (100%), and the best oracle one-bit-per-symbol comparator reaches 75%. These are task countermodels, not installed fly components. The experiment is an **E3 within-family mechanism diagnostic**, not an E4 breadth test or a new model promotion.

## Confirmation results

All values below are means over 64 independent worlds. Intervals shown for the primary and key contrasts are paired, two-sided 95% Student-*t* intervals. Only the first contrast was confirmatory; the others are descriptive and unadjusted for multiplicity.

| Probe | Native choice / contrast | 95% interval |
|---|---:|---:|
| Bout 4, `D=E`, held-out | 58.59% | 54.04–63.14% |
| Bout 8, `Z`, held-out | 46.68% | 42.19–51.17% |
| Bout 8, `E`, held-out | 58.79% | 54.11–63.47% |
| Bout 8, `L`, held-out | 58.40% | 54.16–62.64% |
| Bout 8, `D`, held-out | **60.74%** | 56.62–64.86% |
| **Bout 8, `D−L`, held-out — primary** | **+2.34 pp** | **+0.13 to +4.56 pp** |
| Bout 8, `D−Z`, held-out | +14.06 pp | +7.66 to +20.47 pp |
| Bout 8, `D−L`, *taught* | +3.05 pp | +1.27 to +4.82 pp |
| Day 1, `D−L`, held-out | +2.73 pp | −0.76 to +6.23 pp |
| Final day after new learning, `D_W−L_W`, held-out | +1.76 pp | −1.62 to +5.14 pp |

At bout 8, native taught-pair choice was **77.19%** for `D`, **74.14%** for `L`, and **49.45%** for `Z`. `D−E` held-out choice was +1.95 pp [−2.74, +6.65], and `L−E` was −0.39 pp [−4.88, +4.09]; the schedule comparisons do not establish a robust late-versus-early advantage. The primary world's sign count was 17 positive, 39 tied, and 8 negative, reflecting eight discrete held-out comparisons per world. The held-out signed-margin `D−L` was +0.01615 [0.01150, 0.02080] in native value units; the analog signal is clearer than the thresholded choice effect, but it does not replace the choice endpoint.

The 72-record disjoint new cohort was genuinely learned: at new-cohort end, `D_W−D_S` new-pair choice was **+22.92 pp** [15.67, 30.17], and at the final day **+27.60 pp** [20.40, 34.80]. Yet the new writes harmed old held-out choice under this longer history: at new-cohort end, `D_W−D_S` was **−6.45 pp** [−10.83, −2.07]; at the final day, **−3.13 pp** [−6.02, −0.23]. These are secondary, unadjusted contrasts for this one schedule. In the final `W` branches, `D−L` on **taught** old pairs was +9.69 pp [7.33, 12.04], whereas on never-taught old pairs it was only +1.76 pp [−1.62, 5.14]. More old writing therefore preserves exact taught responses much more clearly than transferable rank choices. The final `D_W−Z_W` held-out advantage remained +11.33 pp [5.75, 16.91], a bounded persistent E3 effect. Cross-study comparison with V9.9B cannot estimate a dose effect because its worlds and sensory duration differ.

## Frozen-code diagnostic and its limit

An external linear ridge probe used only the 20 taught-edge orientation differences in the `Z` branch's frozen 5,177-dimensional KC codes and their taught labels. Its regularizer was fixed before the pilot; held-out labels, hidden ranks, and held-out scores never entered fitting. On the eight withheld edges it scored **82.23%** [78.72, 85.74]; leave-one-*taught-edge*-out choice was **70.08%** [67.10, 73.06]. The `Z` and `D` KC codes agreed to 10⁻¹² in every run's checked probes. These findings show that a particular external supervised linear readout can use information present in the code, even beyond the 75% one-bit-comparator ceiling on this fixture. They do **not** show that REF learned that readout, nor prove that every relevant feature is linearly accessible. The large native–external gap shifts attention toward the native teaching/update/readout interface, without distinguishing those components by itself.

## Decision and audit

The next mechanism experiment should target **how taught pair evidence is accumulated and read across shared symbols**, rather than add more repetitions or expand KC count on the strength of this result. It should retain first-response held-out scoring, a matched no-old-write arm, a final-day new-learning test, and an explicit trained-versus-untrained separation. A structure change or evolution search would be premature until that native interface is localized more tightly. This is not a claim of general ranking, NLP competence, or a fully identified persistent learning core.

The 9903 engineering world passed exposure, source lock, state identity, half-bout equality, predictor equality, read-only, clock, code-separation, and write-ledger checks. The 12-world pilot had no efficacy gate and was excluded from inference. All 64 confirmation worlds completed without optional stopping in **920 seconds**. The independent [audit](results/AUDIT.json) recomputed all 3,264 confirmation metric rows and summary cells, checked the source hashes, and found zero code collisions. The compact bundle is about **1.4 MB** and contains no parent checkpoint or source dataset; see the [manifest](MANIFEST.json), [metric table](results/CONFIRM_METRICS.csv), [summary](results/CONFIRM_SUMMARY.json), and [key figure](results/KEY_FIGURE.png).
