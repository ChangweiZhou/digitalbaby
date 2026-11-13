# Sizing span — report

**Brief.** `OPUS_TASK_SIZING_SPAN_20261113.md`, revision 2, at commit `06cf6b0e` on `codex/minifly-consolidated-20261023`.

**Branch.** `claude/target-test-sizing-stage-1-drmnlq`; nothing went to the default branch.

**Code.** New files only:
- `sz_span.py`, which imports `sz_crosscut` and `sz_learner` without editing them;
- `sz_verify_born.py`.

**Records** are in `runs/span/`:
- one JSON per run, `<block>_<D>_<seed>.json`;
- `admission.json` and `verify_born.json`;
- the tables in `SPAN_TABLES.json`;
- logs in `logs/` and the queue log in `logs_queue.txt`.

`SZ_SPAN_FORECASTS.md` was committed before the first roster run, in `76119087`.

This report gives tables and the mechanical outcome of each registered forecast. **It draws no conclusion about any rule, about the brief's readings, or about the sizing roster.** The 84-fit roster was not run, and neither was anything from the sizing brief's revision 4. No formal checker gate or external review is claimed.

## Admission

All five items passed. Details are in `runs/span/admission.json` and `runs/span/verify_born.json`.

1. **Host.** Python 3.13.18, torch 2.15.0+cpu, Linux x86_64.
2. **Stage 1 reproduction.** Fixed 12, seed 1104, 3,000 chunks. The last-quarter counts equal stage 1's exactly:
   - crossing: 641/643;
   - not crossing: 13,011/13,072.
3. **Mixed stream.**
   - Every repeat byte equals its key at its own D.
   - The one-element sets {12} and {24} give streams byte-identical to stage 1's.
   - Episodes per D in 20,000 chunks: 4 → 57,797; 12 → 58,224; 24 → 58,108; 40 → 58,269.
4. **Births 73129, 93683, 116089.**
   - Each SHA-256 equals its receipt's `weights_sha256`.
   - Each loads strictly into `sz_learner.Net`.
   - The largest logit differences from `arch26.forward` were 5.72e-6, 5.72e-6 and 7.63e-6, against a tolerance of 1e-4.
5. **Permuted start.** Every tensor keeps its sorted values, and its order changed.

**Cost forecast after admission:** 348,420 chunks × 0.1012 s = 9.80 CPU-hours, under the 12-hour cap.

## Choices made where the brief left room

These were fixed in the `sz_span.py` docstring before the roster ran.

- **Classes count all instances of a distance.** Crossing figures are shown beside them.
- **"Too few".** A last tenth with 50% or more right but fewer than 100 instances does not pass. No run hit this.
- **The byte table's padding row** is permuted like every other entry.
- **"Last tenth"** means instances whose repeat byte lies in the last tenth of the stream's bytes.
- **"Repeats to 50%"** means the cumulative repeats of that D at the end of the first 250-chunk window with at least 50% right and at least 100 instances. It is quantized to window ends.
- **Block P births.** Each permuted start is made from the birth paired with its seed.

## Tables

Each table cell gives class, last-tenth right/total (accuracy), crossing right/total, and repeats to 50%. A "—" means the run never reached 50% in a window.

### Block A — random start, fixed D (235×(D+1) chunks)

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 4 | pass 5013/5013 (1.000), cross 78/78, 10,666 | pass 5013/5013 (1.000), cross 78/78, 10,666 | pass 5013/5013 (1.000), cross 78/78, 10,666 |
| 12 | pass 5571/5586 (0.997), cross 261/261, 18,285 | pass 5586/5586 (1.000), cross 261/261, 9,142 | pass 5586/5586 (1.000), cross 261/261, 13,714 |
| 16 | pass 5681/5681 (1.000), cross 355/355, 17,777 | pass 5681/5681 (1.000), cross 355/355, 10,666 | pass 5681/5681 (1.000), cross 355/355, 14,222 |
| 18 | pass 5688/5716 (0.995), cross 353/357, 44,800 | pass 5716/5716 (1.000), cross 357/357, 12,800 | pass 5713/5716 (0.999), cross 357/357, 19,200 |
| 20 | pass 5721/5743 (0.996), cross 448/448, 43,636 | pass 5743/5743 (1.000), cross 448/448, 11,636 | pass 5742/5743 (1.000), cross 448/448, 20,363 |
| 24 | **partial** 748/5785 (0.129), cross 43/542, — | pass 5784/5785 (1.000), cross 542/542, 14,769 | pass 5658/5785 (0.978), cross 530/542, 46,769 |
| 40 | **chance** 239/5873 (0.041), cross 19/917, — | pass 4565/5873 (0.777), cross 715/917, 50,285 | **chance** 237/5873 (0.040), cross 29/917, — |
| 60 | **chance** 230/5919 (0.039), cross 33/1387, — | **partial** 1860/5919 (0.314), cross 399/1387, — | **not held** 210/5919 (0.035), cross 28/1387, 7,225 |

### Block B — random start, mixed D ∈ {4, 12, 24, 40}, 20,000 chunks

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 4 | pass 5823/5823 (1.000), cross 108/108, 1,437 | pass 5714/5714 (1.000), cross 84/84, 1,400 | pass 5762/5762 (1.000), cross 83/83, 1,411 |
| 12 | pass 5881/5881 (1.000), cross 259/259, 1,413 | pass 5868/5868 (1.000), cross 279/279, 1,448 | pass 5769/5769 (1.000), cross 258/258, 1,416 |
| 24 | pass 5768/5768 (1.000), cross 557/557, 1,417 | pass 5862/5862 (1.000), cross 527/527, 1,461 | pass 5803/5803 (1.000), cross 548/548, 1,447 |
| 40 | pass 5828/5828 (1.000), cross 899/899, 1,494 | pass 5790/5790 (1.000), cross 941/941, 1,460 | pass 5852/5852 (1.000), cross 932/932, 1,478 |

### Block C — born start, fixed D

In block C and block P, seed 1104 is paired with birth 73129, seed 1105 with 93683 and seed 1106 with 116089.

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 4 | pass 5013/5013 (1.000), cross 78/78, 32,000 | pass 5013/5013 (1.000), cross 78/78, 21,333 | pass 5013/5013 (1.000), cross 78/78, 21,333 |
| 12 | pass 5567/5586 (0.997), cross 261/261, 27,428 | pass 5571/5586 (0.997), cross 261/261, 13,714 | pass 5586/5586 (1.000), cross 261/261, 13,714 |
| 24 | pass 5275/5785 (0.912), cross 469/542, 34,461 | pass 5752/5785 (0.994), cross 541/542, 12,307 | pass 5783/5785 (1.000), cross 542/542, 17,230 |
| 40 | **chance** 561/5873 (0.096), cross 67/917, — | **partial** 2590/5873 (0.441), cross 383/917, 58,727 | **chance** 263/5873 (0.045), cross 41/917, — |

### Block P — permuted born start, fixed D

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 12 | pass 5586/5586 (1.000), cross 261/261, 9,142 | pass 5586/5586 (1.000), cross 261/261, 9,142 | pass 5586/5586 (1.000), cross 261/261, 9,142 |
| 24 | pass 5770/5785 (0.997), cross 541/542, 12,307 | pass 5780/5785 (0.999), cross 542/542, 7,384 | pass 3253/5785 (0.562), cross 307/542, 7,384 |

### Block D — born start, mixed D ∈ {4, 12, 24, 40}, 20,000 chunks

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 4 | pass 5797/5823 (0.996), cross 108/108, 7,313 | pass 5713/5714 (1.000), cross 84/84, 2,084 | pass 5762/5762 (1.000), cross 83/83, 2,187 |
| 12 | pass 5843/5881 (0.994), cross 258/259, 6,515 | pass 5867/5868 (1.000), cross 279/279, 2,176 | pass 5769/5769 (1.000), cross 258/258, 2,132 |
| 24 | pass 5723/5768 (0.992), cross 552/557, 6,376 | pass 5859/5862 (0.999), cross 527/527, 2,165 | pass 5801/5803 (1.000), cross 548/548, 2,193 |
| 40 | pass 5781/5828 (0.992), cross 890/899, 7,377 | pass 5782/5790 (0.999), cross 941/941, 2,208 | pass 5851/5852 (1.000), cross 932/932, 2,190 |

**Chunk 0**, the first forward pass before any update, had 0 right at every D in all 48 runs. Per-run figures are in each record's `classes.<D>.chunk0`.

## Spans

A block's span is the largest D such that it and every smaller D pass in at least two of three seeds.

| Block | Seeds passing, by D | Span | Next D |
|---|---|---|---|
| A (random, fixed) | 4: 3, 12: 3, 16: 3, 18: 3, 20: 3, 24: 2, 40: 1, 60: 0 | **24** | 40: one seed passes ("mixed" in the brief's wording) |
| B (random, mixed) | 4: 3, 12: 3, 24: 3, 40: 3 | **40** (largest D in the block) | — |
| C (born, fixed) | 4: 3, 12: 3, 24: 3, 40: 0 | **24** | 40: none pass |
| P (permuted, fixed) | 12: 3, 24: 3 | **24** (largest D in the block) | — |
| D (born, mixed) | 4: 3, 12: 3, 24: 3, 40: 3 | **40** (largest D in the block) | — |

## Forecasts

The forecasts are copied unchanged in `SZ_SPAN_FORECASTS.md`. Outcomes were read mechanically from the tables above.

| | Forecast | Registered | Outcome |
|---|---|---|---|
| F1 | Block A passes at 18 in ≥ 2 of 3 | about 50% | **held**: 3 of 3 |
| F2 | Block A passes at 24 in ≥ 2 of 3 | about 40% | **held**: 2 of 3 (1105 and 1106; 1104 is partial) |
| F3 | Block A passes at 60 in ≥ 2 of 3 | about 15% | **did not hold**: 0 of 3 |
| F4 | Block B passes at 24 or 40 in ≥ 2 of 3 where block A does not | about 25% | **held** at D = 40: B passes 3 of 3, A passes 1 of 3. At 24 both pass in ≥ 2 of 3, so 24 does not qualify |
| F5 | At D = 12, repeats to 50% are more from born than from random in ≥ 2 of 3 pairs | about 75% | **held**: 2 of 3 |
| F6 | Block C passes at 40 in ≥ 2 of 3 | about 20% | **did not hold**: 0 of 3 |

F5's three pairs (born against random) were:
- 1104: 27,428 against 18,285;
- 1105: 13,714 against 9,142;
- 1106: 13,714 against 13,714. This is a tie, which is not "more".

## Figures for the brief's other comparisons

These are given without a reading.

**Mixed against fixed, random start: repeats to 50%, B against A.**
- At D = 24:
  - 1104: 1,417 against — (A partial);
  - 1105: 1,461 against 14,769;
  - 1106: 1,447 against 46,769.
- At D = 40:
  - 1104: 1,494 against — (A at chance);
  - 1105: 1,460 against 50,285;
  - 1106: 1,478 against — (A at chance).

**Born against random against permuted, fixed D: repeats to 50%, C / A / P.**

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 4 | 32,000 / 10,666 / (not in P) | 21,333 / 10,666 / — | 21,333 / 10,666 / — |
| 12 | 27,428 / 18,285 / 9,142 | 13,714 / 9,142 / 9,142 | 13,714 / 13,714 / 9,142 |
| 24 | 34,461 / — / 12,307 | 12,307 / 14,769 / 7,384 | 17,230 / 46,769 / 7,384 |
| 40 | — / — / (not in P) | 58,727 (partial) / 50,285 / — | — / — / — |

**Born against random, mixed D: repeats to 50%, D / B.**
- 1104: 6,376 to 7,377 against 1,413 to 1,494;
- 1105: 2,084 to 2,208 against 1,400 to 1,461;
- 1106: 2,132 to 2,193 against 1,411 to 1,478.

## CPU

| Block | Runs | Chunks | CPU-s (generation + training) | CPU-s per chunk (range) |
|---|---|---|---|---|
| A | 24 | 142,410 | 13,975 | 0.0927–0.1009 |
| B | 3 | 60,000 | 5,960 | 0.0991–0.0994 |
| C | 12 | 59,220 | 5,922 | 0.0924–0.1022 |
| P | 6 | 26,790 | 2,613 | 0.0942–0.0990 |
| D | 3 | 60,000 | 5,865 | 0.0974–0.0980 |
| **Total** | **48** | **348,420** | **34,335 (9.54 CPU-hours)** | mean 0.0985 |

- The forecast after admission was 9.80 CPU-hours.
- Wall time was 18:09:34 to 21:23:00 (3 h 13 min), with three processes and one thread each, on one Linux x86_64 host.
- All 48 runs exited with code 0. Admission is not included above; its main item (stage 1 reproduction) took 303.7 CPU-s.

## What was unclear

1. **Two runs collapsed after holding near 100%.** The brief did not anticipate this, and I did not look into the cause. These are the only two runs, out of all 48 runs and every distance, in which a window at 90% or more was followed by one below 70%.
   - **A 60, seed 1106.** Per-window accuracy rose to 95–100% by window 9 and stayed between 71% and 100% through window 48. It fell to 13% in window 49 and stayed at 2–4% through window 58, the end. The run is classed "not held" and counts as not passing. The brief's "not held" class covers it, but not the shape: about 40 windows at near 100% at D = 60, then chance. Block A's span (24) and F3 would be the same had it held: with it, D = 60 passes in 1 of 3.
   - **P 24, seed 1106.** It held at 97–100% from window 5 to window 22, fell to 4% in window 23, and was at 85% in window 24, the last. Its last tenth (0.562) straddles the drop, so it is classed *pass*. A last tenth placed slightly differently would have classed it otherwise.
2. **Identical figures across seeds.** Many fixed-D rows are the same in all three seeds, for example A at D = 4 (10,666 each), P at D = 12 (9,142 each), and C 12 / C 4 in two of three seeds.
   - For fixed D, the episode layout and therefore the instance counts per window are fixed by D. Only the keys and filler differ by seed.
   - Repeats to 50% is quantized to window ends: at D = 12 one window is about 4,571 repeats.
   - Equal figures therefore mean "reached 50% in the same window", and F5's tie in pair 1106 is a tie at that granularity.
3. **Repeats to 50% does not track the last-tenth class.**
   - C 40 / 1105 first reached 50% in its last window (58,727 = all its repeats) but is at 44.1% over the last tenth, so it is *partial*.
   - A 60 / 1106 reached 50% at 7,225 repeats and is *not held*.
4. **Runs still rising at the end.** I did not extend any stream (that would be NOT_RUN), so these classes are at the brief's length only. Per-window accuracy in the last windows:
   - A 24 / 1104: 7%, 14%, 17%;
   - A 60 / 1105: 25–37%;
   - C 40 / 1104: 6–9%.
5. **Equal dose between mixed and fixed.** The brief compares B and A at about 60,000 repeats per D. This held: B ≈ 58,000 per D and A ≈ 57,800–58,700.
   - B's repeats arrive at four distances at once, about 4× the per-update repeats of a fixed stream at D = 24.
   - The brief's comparison does not separate this, and I report it without a reading.
6. **Recording.** Window records keep crossing and not-crossing counts per D. Quarter figures, which the stage 1 code also supports, were not used for any class.

## Not run

- The 84-fit sizing roster.
- Anything from the sizing brief's revision 4.
- Distances of hundreds of bytes (lines), which are out of this brief.
- Any extension of a stream beyond its briefed length.
- Any investigation of the two collapses.
