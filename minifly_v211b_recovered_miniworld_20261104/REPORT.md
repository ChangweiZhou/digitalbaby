# V211B · Fresh-seed run of the recovered MiniWorld implementation

**Disposition** Positive development replication on the original MiniWorld protocol. The Round 16 executable was extracted from the session export, with its two subsequent logged source edits. The recovered code was not modified for this run. The world-seed range changed from 0–9 to 10–19. Python hash seed stayed at 1 to isolate the world-seed change. Each experienced learner visited 30 earlier worlds with 40 questions per world. Each target panel had 40 questions per learner.

| Feedback mode | Experienced | Fresh | First 10 experienced / fresh | Mean extra correct per 40 (reported 95% half-width) | Archived extra correct |
|---|---:|---:|---:|---:|---:|
| True | 98.2% | 92.8% | 98% / 75% | +2.2 ± 0.70 | +4.0 ± 1.17 |
| Random location from current world | 83.5% | 66.2% | 79% / 38% | +6.9 ± 2.27 | +6.4 ± 1.66 |
| None | 4.8% | 7.5% | 5% / 12% | −1.1 ± 1.66 | −0.2 ± 1.09 |

The intervals use the original script's normal approximation on ten paired worlds. All three source rows printed. The macOS timing wrapper returned an error reading the kernel clock-rate statistic after the run and after printing 33.71 seconds of wall time. No trajectories were repeated to collect timing. The scheduled census was 36,000 prior and 2,400 target question applications.

## Scope and implementation checks

The two learners in a feedback mode get the same target schedule and target feedback policy. Across feedback modes the target feedback also changes. The random mode draws a place from the current world and may draw the true place. These comparisons replicate the original development assay. They cannot isolate the causal contribution of prior correct answer assignment under identical target feedback. The separately drafted V211 protocol does require that match.

All target templates come from a disjoint template pool. The vocabulary generator draws independently from a small syllable set without enforcing prior–target disjointness. The dry exposure audit records overlaps in the declared pools, including case-sensitive and case-folded counts. Treat this as fresh worlds with unseen exact templates and partly overlapping vocabulary. It does not certify strict jointly novel lexical transfer.

Round 15's separate validation learner was also recovered by applying its recorded edits to the pinned Round 14 source. With the original dataset relocated in memory only, it exactly reproduced the archived base-validation row at hash seed 7 (100.0% accuracy, 0.1% ties). That checks the recovered parent on one historical row. The V211 four-arm continuous-life design was not executed here.

## Reproduction

From this directory, using Python with NumPy installed:

```sh
env K=30 NQ=40 NT=40 NS=10 S0=10 PYTHONHASHSEED=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -B miniworld.py
```

Recovery and exposure scripts, source hashes and stdout accompany this report. The session export and raw QA corpus are outside the bundle.
