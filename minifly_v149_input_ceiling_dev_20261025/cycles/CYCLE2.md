# Cycle 2: dev test run on Buster Bear (25 Oct 2026)

Full protocol size: 40 KB taught, 5 KB calibration, 10 KB held-out. Wide grids were used to fix the settings. **Development result**; nothing was scored on the main text.

Command: `python run.py --text buster --train 40 --calib 5 --held 10 --l2 300,100,30,10,3,1 --maxiter 300 --tag c2_dev`. Output: `results/c2_dev/` and `results_c2_dev.log`.

## Results (held-out bits per symbol)
Counting ladder: unigram 4.497, bigram 3.510, PPM-2 2.870, PPM-3 2.527, PPM-4 2.450.

**Readout checks** (R1 on evaluator-side codes). All three pass at full size:

| Code | R1 | Target | Check |
|---|---:|---|---|
| last symbol, one-hot | 3.524 | bigram 3.510 | within 0.15: PASS |
| last two symbols, one-hot | 2.834 | PPM-2 2.870 | within 0.20: PASS |
| constant | 4.492 | unigram 4.497 | not below unigram − 0.02: PASS |

The smoothing mismatch seen in the cycle-1 tiny run (0.06 bits at 4 KB) shrinks to 0.005 bits at 40 KB. The floor check is kept unchanged.

**Arms:**

| Arm | R1 | R1 accuracy | R1 chosen L2 | R2 |
|---|---:|---:|---:|---:|
| PERM-obs (primary) | 3.718 | 0.300 | 30 | 4.313 |
| PERM-x | 3.764 | 0.296 | 30 | 4.323 |
| FE0-x | **3.184** | 0.407 | 10 | 4.195 |
| PERM-last | 3.513 | 0.280 | 10 | 3.774 |

* Line-bootstrap SDs (1,000 resamples of held-out lines) are 0.024–0.026 for the three contrasts the predictions use. The 0.10 margin is about four SDs.
* Applying the registered rules to this dev run gives CONTEXT_FIRST and all three predictions as predicted:
  * C = 3.718 against B − 0.10 = 3.410;
  * PERM-last is within 0.003 of bigram, and PERM-x is worse than PERM-last;
  * R2 is 0.60 worse than R1.

  These are dev numbers only.

## Reading (development, not evidence)
* **The V148 front end's state carries less than bigram-level context.** Its full trace (PERM-x, 3.76) is *worse* than the last byte alone (PERM-last, 3.51): the slowly decaying, permuted trace blurs the most recent byte. Adaptation (PERM-obs) changes little.
* **FE0 carries more than bigram-level context on this text: 3.18, 0.33 better than bigram.** This is despite `fastmem.py` describing it as keeping "almost no order". It decays faster (0.81 per byte, against 0.95), so recent bytes dominate. This observation is informed by the dev data. It is recorded here and in the lock as a dev-informed expectation, **not** as a prediction, and the registered decision rule (PERM-obs) is unchanged.
* **R2 (the KC vote) stays near unigram for every arm.** Averaging single-unit distributions loses most of what R1 finds.

## Settings fixed here (`cycles/SETTINGS.json`), before the rehearsal
1. **L2 grid `100,30,10,3,1`.**
   * 300 was never chosen.
   * The arms chose 30 or 10.
   * The one-hot check codes chose 1, so the low end stays.
2. **R2 grid `0.001,0.003,0.01,0.03,0.1`.** Every arm chose 0.01, the lowest value tried, so the grid is extended downwards.
3. **Iteration limit 300.** Many arm fits stopped at 300 without the L-BFGS convergence flag. A cold-start check (`cycles/converge_check.py`, `converge_check.json`) refitted the primary arm (L2 30) and FE0-x (L2 10) to 900 iterations. Calibration and held-out bits changed by less than 0.001 in both:

   | Arm | 300 iterations (calibration / held-out) | 900 iterations |
   |---|---:|---:|
   | PERM-obs | 3.50901 / 3.71724 | 3.50897 / 3.71768 (converged at 592) |
   | FE0-x | 2.79985 / 3.18314 | 2.79965 / 3.18391 |

4. **Unchanged:** sizes 40 / 5 / 10 KB, the readout checks, the decision rule (margin 0.10) and the three predictions.

## Cost
* States: 298 CPU-s.
* The whole run took 4,629 CPU-s; the arms' R1 fits took 940–1,110 CPU-s each. Peak memory 1.4 GB.
* With the narrower grid, a run is expected to take about 1.1 CPU-hours.
* The convergence check took another 0.5 CPU-hours.
