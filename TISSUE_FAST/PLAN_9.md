# Plan 9: plan 8's best on two more seeds, and a cap on connections at totals 1 (21 November 2026)

Written before its runs. Development level. **Authorisation:** owner, 21 November 2026 ("继续做"; "要的就是快速迭代"). **Program:** `TISSUE_EVENT/` at this commit; the second run of every tissue without records is now off by default (`--verify-core` turns it on), which halves the time; nothing else changed.

**Arms** (all with turnover, trial threshold 1, making period 32,768, withdrawal by use, 16,384 bytes; first form C_E = C_I = 1, trial 2,048; coincidence form C_E 1, C_I 1.5, trial 512):
- **Seeds:** plan 8's first-form B10 and coincidence-form B10i on seeds 93683 and 116089 (4 tissues).
- **Cap:** the same two settings on seed 73129 with `--cap` 4, 6, 8, 12 (incoming connections, lines included) (8 tissues).

**H9a.** On both new seeds both settings are at level at d = 8. **H9b.** Some cap carries the byte farther than no cap (plan 8: d = 8 in both forms). Refuted case by case.

| No. | Prediction | Chance |
|---|---|---|
| K1 | H9a holds in all four cases | 0.4 |
| K2 | H9b holds in some form | 0.35 |
| K3 | Cap 4 is worse than cap 8 in both forms | 0.6 |

**Cost:** 12 tissues at about 40 s each, 3 rounds on 4 cores: about 3 minutes.

## Outcome

(to be entered after the runs)
