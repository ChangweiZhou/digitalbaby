# V223 — official staged development run

**2026-11-05 · Completed all locked A doses.**

The three-world acquisition screen selected the connected width-32 GRU with direct byte input. The main comparison used exact forward sensitivities and BPTT32 on the same graph, objective, exposures, and paired births.

At the final A dose, exact forward credit recalled **20/192** facts and answered **7/1,536** held sums. BPTT32 recalled **18/192** facts and answered **6/1,536** held sums. These totals pool items for readability. The paired worlds are the independent units.

## Acquisition screen

Cells give exact fact and arithmetic answers with own EOS, out of four each. Every arm received 256 presentations of each record.

| Arm | 66101 facts / arithmetic | 66202 facts / arithmetic | 66203 facts / arithmetic |
|---|---:|---:|---:|
| dense64 | 4/4 · 4/4 | 4/4 · 4/4 | 4/4 · 4/4 |
| connected32 | 4/4 · 4/4 | 4/4 · 4/4 | 4/4 · 4/4 |
| connected64 | 4/4 · 4/4 | 4/4 · 4/4 | 4/4 · 4/4 |
| isolated | 4/4 · 4/4 | 4/4 · 4/4 | 4/4 · 4/4 |
| kc | 2/4 · 0/4 | 2/4 · 2/4 | 1/4 · 2/4 |

The direct-input matched isolated and connected masks tied on all three final fixture scores. KC input missed the fact floor. The fixture supports acquisition on the direct-byte interface and leaves a connectivity advantage on more demanding streams open.

## Main dose curves

Each cell gives exact initial fact counts on the three fresh worlds in order. The primary arithmetic panel contains 512 held questions per world.

| Addition dose | Forward facts | BPTT32 facts | Forward held arithmetic | BPTT32 held arithmetic |
|---:|---:|---:|---:|---:|
| 4,096 | 1/64 · 0/64 · 0/64 | 1/64 · 0/64 · 0/64 | 0/512 · 0/512 · 0/512 | 0/512 · 0/512 · 0/512 |
| 8,192 | 0/64 · 0/64 · 0/64 | 0/64 · 0/64 · 0/64 | 0/512 · 0/512 · 0/512 | 0/512 · 0/512 · 0/512 |
| 16,384 | 0/64 · 0/64 · 0/64 | 0/64 · 0/64 · 0/64 | 0/512 · 1/512 · 5/512 | 0/512 · 1/512 · 3/512 |
| 32,768 | 2/64 · 2/64 · 1/64 | 1/64 · 1/64 · 1/64 | 5/512 · 2/512 · 2/512 | 2/512 · 2/512 · 2/512 |
| 65,536 | 11/64 · 5/64 · 4/64 | 10/64 · 4/64 · 4/64 | 0/512 · 0/512 · 7/512 | 0/512 · 1/512 · 5/512 |

The locked A gate required at least 52/64 facts on two reference worlds. **0/3 reference worlds cleared it.**

## Final A behavior and shortcuts

| World | Arm | Taught | Held | Longer operands | Both carry answers | Held first byte | Held bits/answer byte | Held EOS under true prefix |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 66301 | forward | 0/128 | 0/512 | 0/256 | 0/128 | 375/512 | 2.559 | 512/512 |
| 66301 | bp | 0/128 | 0/512 | 0/256 | 0/128 | 351/512 | 2.605 | 512/512 |
| 66302 | forward | 1/128 | 0/512 | 0/256 | 0/128 | 386/512 | 2.523 | 512/512 |
| 66302 | bp | 2/128 | 1/512 | 0/256 | 0/128 | 385/512 | 2.524 | 512/512 |
| 66303 | forward | 0/128 | 7/512 | 0/256 | 0/128 | 383/512 | 2.483 | 512/512 |
| 66303 | bp | 0/128 | 5/512 | 0/256 | 0/128 | 375/512 | 2.503 | 512/512 |

| World | Frequency exact held | Nearest exact held | No-carry exact held |
|---:|---:|---:|---:|
| 66301 | 0/512 | 31/512 | 94/512 |
| 66302 | 1/512 | 29/512 | 94/512 |
| 66303 | 1/512 | 33/512 | 90/512 |

The latest-value fact reference retrieves the exact exposed label for every taught fact. The arithmetic shortcuts are fitted on the A exposure stream and use explicit EOS on their complete outputs. The both-correct and length-extrapolation shortcut counts are saved in the per-world A receipts.

## Interference and correction

B and A′ were skipped under the locked acquisition rule. The receipts establish A-phase learning behavior. Retention and revision under adequate acquisition remain untested.

## Cost and evidence scope

The selected graph was profiled over 8,192 symbols per method before opening the main worlds. Throughput was 10,229 forward and 7,902 BPTT symbols per CPU second. The planned training and probe allocation with reserve was 17.7 CPU minutes.

Recorded worker CPU was 1.4 minutes for screening, 13.1 for A and 0.0 for B/A′. These receipts exclude separate arithmetic-shortcut calculation and supervisor overhead. Per-worker peak RSS is recorded in the run receipts.

The comparison uses three paired development worlds. The exact-forward learner begins at random birth and computes its own updates. Its finite-window gradient and the reverse diagnostic target the same computation, with numerical differences possible over long runs. This campaign supplies no independent sealed confirmation.

Run `screen.py`, then `official.py` and `report_official.py` using the single-thread environment from CYCLES.md. The output directory contains locks, raw native outputs, phase checkpoints and the decision receipt.
