# E31 stage 2, worker A: G, H and P2b, three seeds each (14 November 2026)

**Status.** Nine complete lives. This is E2 evidence, from single lives. The receipts are `exp45/e31/e31_life_<arm>_<seed>.json`, with their weights; each life was pushed as it ended. The table is `E31_TABLE_A.json`, made by `e31_table_A.py` from the receipts only.

- **Brief.** `OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md` with addenda 2 and 3.
- **Code.** `exp31_switch.py` and `arch31.py`, as committed at `1ea99869`.
- **Host.** Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, one thread a life, three at a time.

## Checks on every receipt

- The weights file matches the receipt's `weights_sha256`.
- 36,864 optimizer steps, and every loss is finite.
- The data digest equals the floor's (`e26_life_S_<seed>`).
- The birth was admitted with the recorded `exp16_integrated.py` exception (addendum 2, ruling 5).
- CPU time: G 1.29–1.33 h, H 1.21–1.27 h, P2b 1.75–1.93 h.

## End of life (stage 6), percent whole answers

Measures are those of `exp45/e26/verify_lives.py`:
- **within:** new instances of taught pairs (600 items);
- **plus-minus:** `+-` and `-+` pooled (1,200);
- **F1 and F3:** family means at k = 12.

| arm | seed | within | plus-minus | F1 | F3 | U (P2b) |
|---|---|---|---|---|---|---|
| floor (E26 life S) | 73129 | 83.5 | 54.2 | 95.7 | 89.2 | |
| | 93683 | 83.8 | 44.5 | 94.6 | 91.9 | |
| | 116089 | 81.0 | 37.2 | 96.0 | 95.1 | |
| ceiling (E26-X life SB) | 73129 | 93.2 | 72.3 | 98.5 | 97.6 | |
| | 93683 | 97.3 | 82.8 | 99.1 | 98.5 | |
| | 116089 | 93.0 | 73.9 | 98.4 | 98.2 | |
| **G** | 73129 | 84.7 | 56.7 | 96.8 | 93.1 | |
| | 93683 | 84.7 | 55.6 | 96.8 | 93.5 | |
| | 116089 | 87.0 | 65.6 | 97.4 | 95.9 | |
| **H** | 73129 | 86.7 | 72.9 | 96.8 | 94.3 | |
| | 93683 | 91.7 | 74.5 | 98.4 | 95.7 | |
| | 116089 | 92.2 | 71.4 | 97.0 | 96.1 | |
| **P2b** (λ 4.461 / 5.501 / 4.711) | 73129 | 82.5 | 50.0 | 94.7 | 91.6 | 38.0 |
| | 93683 | 91.8 | 73.6 | 98.1 | 92.9 | 18.2 |
| | 116089 | 78.2 | 45.1 | 95.9 | 92.4 | 41.5 |

## Outcomes resting on worker A's arms only

| | figures by seed (73129 / 93683 / 116089) | outcome |
|---|---|---|
| **P31.1** G at least 15 points above the floor in two of three seeds | +2.4 / +11.1 / +28.3 | **misses** |
| **P31.2** H within 10 points of the ceiling in two of three seeds | ceiling − H: −0.6 / 8.3 / 2.5 | **holds** (all three) |
| **P31.7** P2b's end-of-life U between 35% and 65% in all three seeds | 38.0 / 18.2 / 41.5 | **misses** |

**Fixed readings for G and H.** The rule is whether an arm closes at least half the floor-to-ceiling gap in two of three seeds.
- G closes half the gap in 1 of 3 seeds (116089).
- H closes half the gap in 3 of 3.
- The brief's fixed reading for "H does and G does not": *the recurrent pathway as a whole is* the bottleneck.

The brief does not read differences under 15 points between single lives. Two of G's three gains over the floor (2.4 and 11.1) are under 15.

P2b's gains over the floor are −4.3, +29.1 and +7.8. They enter P31.8 and P31.9, which also need P2a and P2c.

**Not judged here, because they need worker B's lives:** P31.3 and P31.4 (K), P31.8 and P31.9 (P2a, P2c), and P31.10 (Ks). The combined `E31_REPORT.md` waits for those lives.
