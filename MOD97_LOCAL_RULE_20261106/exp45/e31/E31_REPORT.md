# E31 report: lives under the sweep with one switch each (worker B, 15 November 2026)

**Brief:** `OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md` with addenda 2 and 3. This is E2 evidence, from single lives.

**Code:** `exp31_switch.py` and `arch31.py`. The lives ran in a work folder holding `exp16_integrated.py` as at the E26 births (blob 774813cf), the exception recorded under addendum 2, ruling 5.

**Table:** made by `e31_report_B.py` from receipts only, written to `E31_TABLE.json`. It reads worker A's receipts from their branch at a930ed95 and recomputes them independently; the result equals worker A's `E31_TABLE_A.json` to every figure.

## What ran

| Worker | Lives | Branch |
|---|---|---|
| A | G, H, P2b × 3 seeds | `claude/sleepy-knuth-q96dd1` at a930ed95 |
| B | K, P2a, P2c × 3 seeds | `claude/charming-clarke-u9g850` |

Eighteen complete lives. Each passes these checks:
- 36,864 optimizer steps;
- weights equal to the receipt's hash;
- the data digest equal to the floor's (`e26_life_S_<seed>`);
- all losses finite.

Worker B's CPU per life: K 1.19–1.20 h, P2a 1.92–1.93 h, P2c 1.87 h.

**Not run:**
- **Arm Ks:** not run at full length. Addendum 3 released it, but the user told worker B not to run it at full length. Its stage-1b dry life is on file.
- **Arm P:** withdrawn before any full life (addendum 2, ruling 3).

**Restart:** a container reset at about 00:45 UTC on 15 November killed the first attempt at the three P2a lives 10–20 minutes in. They were rerun from their start; the killed runs' logs are in `logs_b/restart_killed/`.

**λ on the first batch:**

| Arm | Ratio | 73129 | 93683 | 116089 |
|---|---|---|---|---|
| P2a | 0.25 | 22.307 | 27.507 | 23.555 |
| P2b | 0.05 | 4.461 | 5.501 | 4.711 |
| P2c | 0.01 | 0.892 | 1.100 | 0.942 |

## End of life (stage 6), percent whole answers

Measures are those of `exp45/e26/verify_lives.py`:
- **within:** new instances of taught pairs (600 items);
- **plus-minus:** `+-` and `-+` pooled (1,200);
- **F1 and F3:** family means at k = 12.

U is the end-of-life undecided share on the plus-minus panel (E30's definition), for the P2 arms. The cosine is between K's `Bhh` and the forward recurrent matrix.

| Arm | Seed | Within | Plus-minus | F1 | F3 | U / cosine |
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
| **K** | 73129 | 88.7 | 74.4 | 97.2 | 91.4 | cos 0.99999999 |
| | 93683 | 92.7 | 78.2 | 98.8 | 94.3 | cos 0.99999999 |
| | 116089 | 85.8 | 63.9 | 96.8 | 94.6 | cos 0.99999999 |
| **P2a** (0.25) | 73129 | 78.3 | 48.2 | 91.7 | 87.6 | U 28.3 |
| | 93683 | 72.5 | 33.2 | 91.7 | 87.2 | U 21.7 |
| | 116089 | 73.0 | 34.7 | 93.8 | 90.2 | U 29.3 |
| **P2b** (0.05) | 73129 | 82.5 | 50.0 | 94.7 | 91.6 | U 38.0 |
| | 93683 | 91.8 | 73.6 | 98.1 | 92.9 | U 18.2 |
| | 116089 | 78.2 | 45.1 | 95.9 | 92.4 | U 41.5 |
| **P2c** (0.01) | 73129 | 84.2 | 44.9 | 96.4 | 93.2 | U 50.4 |
| | 93683 | 86.2 | 47.6 | 95.6 | 92.6 | U 44.3 |
| | 116089 | 83.0 | 51.1 | 97.3 | 95.2 | U 56.2 |

Plus-minus above the floor, by seed (73129 / 93683 / 116089):

| Arm | Above floor |
|---|---|
| G | +2.4 / +11.1 / +28.3 |
| H | +18.8 / +30.0 / +34.2 |
| K | +20.2 / +33.8 / +26.7 |
| P2a | −6.1 / −11.3 / −2.6 |
| P2b | −4.3 / +29.1 / +7.8 |
| P2c | −9.3 / +3.1 / +13.8 |

## Outcomes

| | Prediction | Chance | Figures (73129 / 93683 / 116089) | Outcome |
|---|---|---|---|---|
| P31.1 | G at least 15 points above the floor in at least two of three seeds | 45% | +2.4 / +11.1 / +28.3 | **misses** |
| P31.2 | H within 10 points of the ceiling in at least two of three seeds | 35% | ceiling − H: −0.6 / 8.3 / 2.5 | **holds** (all three) |
| P31.3 | K within 5 points of H in all three seeds | 60% | K − H: +1.5 / +3.8 / −7.5 | **misses** (seed 116089) |
| P31.4 | K's final cosine between `Bhh` and the forward matrix above 0.99 in all three | 85% | 0.99999999 in each | **holds** |
| P31.5, P31.6 | (arm P) | | | withdrawn unjudged (addendum 2, ruling 3) |
| P31.7 | at strength 0.05 the end-of-life U lies between 35% and 65% in all three seeds | 40% | 38.0 / 18.2 / 41.5 | **misses** (seed 93683) |
| P31.8 | at least one strength 10 points or more above the floor on plus-minus in at least two of three seeds | 35% | P2a 0 seeds, P2b 1, P2c 1 | **misses** |
| P31.9 | the strongest push is not the best of the three in at least two of three seeds | 60% | best: P2b / P2b / P2c | **holds** (all three) |
| P31.10 | Ks within 5 points of H; Ks above K at F1.2 | 60%, 65% | — | unjudged: Ks not run at full length |

Three of eight judged predictions hold: P31.2, P31.4 and P31.9.

## Readings (the brief's fixed sentences)

**Closing half the floor-to-ceiling gap on plus-minus**, in two of three seeds:

| Arm | Seeds where it closes half the gap |
|---|---|
| G | 1 of 3 (116089) |
| H | 3 of 3 |
| K | 3 of 3 |
| P2a | 0 |
| P2b | 1 (93683) |
| P2c | 0 |

H does and G does not: *the recurrent pathway as a whole is* the bottleneck.

**K is read only against H.** Its differences from H are +1.5, +3.8 and −7.5 points. All are under 15 points between single lives, so they are not read.

**P2.** By P31.8 no strength clears 10 points over the floor in two seeds. Any strength that had would have been run again on fresh seeds before a claim. The brief fixes no further reading.

Two of G's gains over the floor (2.4 and 11.1) are under 15 points and are not read.
