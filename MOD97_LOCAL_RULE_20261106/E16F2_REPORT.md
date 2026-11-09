# E16-F2 report: the label-free reservoir at 65,536 on the fresh seeds (8–9 November 2026)

Brief: `OPUS_TASK_E16F2_RESERVOIR_FRESH_SEEDS_20261108.md`, as amended at 17:25 EST, run alongside the E16-F reruns as the E16-F amendment of 17:45 EST directs. Predictions: `E16F2_PREDICTIONS.md`, written and pushed before launch. Arm: `E16_REPLAY=total E16_CAP=65536 python3 exp16_integrated.py T <seed>`, on E16-F's births. The twin is E16-F's `exp45/e16_life_D_<seed>.json`. Tables: `exp45/SUMMARY_E16F2.md`.

**Outcome:** all three predictions hold in every fresh seed. One reservoir of 65,536 entries, with no family label anywhere in the replay path, keeps the learner at 0.82–0.99 of the twin on every family.

## Verdicts

Scores are family means at k = 12 for operations and the mean final-answer accuracy over the four held pairs for compositions, as in E16-F.

### P16F2.1: ratio to the twin at least 0.80 on all three families, holds in every seed

| Seed | mod-97 ops: learner / twin | Held comp: learner / twin | mod-73 ops: learner / twin | Verdict |
|---|---|---|---|---|
| 200003 | 91.9 / 94.9 = 0.97 | 80.5 / 93.7 = 0.86 | 91.6 / 92.6 = 0.99 | holds |
| 300007 | 89.0 / 94.1 = 0.95 | 79.8 / 91.5 = 0.87 | 92.0 / 92.8 = 0.99 | holds |
| 400009 | 89.2 / 95.3 = 0.94 | 77.0 / 94.0 = 0.82 | 91.0 / 94.3 = 0.96 | holds |

The standing seeds gave 0.97 / 0.96 / 0.88 on operations, 0.88 / 0.81 / 0.83 on compositions and 0.99 / 0.98 / 0.95 on later operations.

### P16F2.2: composition loss from block 4 to 6 at most 15 points, and distinct count at block 5 between 27,500 and 29,500, holds in every seed

| Seed | Held comp, block 4 to 6 | Loss | Distinct entries at block 5 | Verdict |
|---|---|---:|---:|---|
| 200003 | 89.3 to 80.5 | 8.8 | 28,431 | holds |
| 300007 | 87.8 to 79.8 | 8.0 | 28,334 | holds |
| 400009 | 88.8 to 77.0 | 11.8 | 28,298 | holds |

The standing seeds lost 5.4 / 11.9 / 11.8 points and held about 28,400 distinct entries. At block 5 the reservoir's composition by family, recorded for reading only, was 9,134–9,167 mod-97 entries, 14,472–14,613 compositions and 4,651–4,687 mod-73 entries.

### P16F2.3: acquisition, holds in every seed

| Seed | mod-97 ops after block 2, at least 85 | Held comp after block 4, at least 80 | Verdict |
|---|---:|---:|---|
| 200003 | 91.6 | 89.3 | holds |
| 300007 | 93.1 | 87.8 | holds |
| 400009 | 88.0 | 88.8 | holds |

## Against the every-entry arm on the same births

E16-F ran the every-entry, family-balanced arm on the same three births and against the same twin, so the two arms can be compared seed by seed.

| Seed | mod-97 ops after 6: every entry / reservoir | Held comp after 6 | Comp loss, block 4 to 6 | mod-73 ops after 6 |
|---|---|---|---|---|
| 200003 | 90.3 / 91.9 | 83.3 / 80.5 | 5.4 / 8.8 | 91.3 / 91.6 |
| 300007 | 86.4 / 89.0 | 77.0 / 79.8 | 12.0 / 8.0 | 90.9 / 92.0 |
| 400009 | 84.4 / 89.2 | 81.5 / 77.0 | 8.2 / 11.8 | 92.1 / 91.0 |

- **On the mod-97 operations, the reservoir is better in every seed,** by 1.6 to 4.8 points. In seed 400009 its dip during the composition blocks is smaller: 79.5 at worst against 71.5.
- **On compositions, the two arms are level within a few points,** and neither is better in every seed. Seed 300007, the every-entry arm's one miss, keeps more under the reservoir.
- **On the mod-73 operations, the two arms are equal within about one point.**

This is three seeds and one stream, so the differences are descriptive, not tested.

## Machine and pace

| Item | Value |
|---|---|
| Machine | as E16-F: 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Harness | commit 1857c49d, which contains 72d3aca and 7703d6c; no harness file changed by this job |
| CPU per life | 5,088 / 5,079 / 5,116 s |
| Wall | 22:37:13 to 00:38:00–00:40:35 UTC, about 2 h 02 min; six processes on four cores until 00:26, then three |
| Forecast | 2.9–3.5 h of wall; the lives finished in about 2 h, because the E16-F reruns ended earlier than assumed and freed their cores |

## Notes

- **Launch:** each life was detached with `setsid nohup`, as the E16-F amendment directs. Nothing was killed or restarted. The background waiter that watched these lives hit its own two-hour limit at about 00:37 UTC; it only read the logs, the lives were unaffected, and a new waiter saw them exit 0.
- **Logs:** the logs were written to the session's scratch folder and copied to `exp45/logs/e16f2_tot65536_<seed>.log` at the end, so that no `git pull --rebase` could cut them off.
- **Settings record:** the receipts' settings record shows `replay16: total` and `cap: None`. The reservoir itself held 65,536 entries at every block, as each receipt's `buffer` field shows, so the run used the intended capacity. Only the record of it is missing. The harness was not changed.
- **Shared table:** `exp45/SUMMARY_E16F2.md` holds the E16-F rows, every-entry arm and twin, beside this arm's, because the summarizer reads every arm present for these seeds.

## Files

- **Predictions:** `E16F2_PREDICTIONS.md`.
- **Receipts:** `exp45/e16_life_T_tot65536_{200003,300007,400009}.json`.
- **Logs:** `exp45/logs/e16f2_tot65536_<seed>.log`, `exp45/logs/e16f2_lives_wall.txt`.
- **Tables:** `exp45/SUMMARY_E16F2.md`.
- **Review paragraph:** `E16F2_REVIEW_SECTION_9_36.md`.
