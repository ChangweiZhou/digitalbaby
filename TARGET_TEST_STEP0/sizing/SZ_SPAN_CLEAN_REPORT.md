# Span with clean bytes: report

**Brief:** `OPUS_TASK_SIZING_SPAN_CLEAN_ALPHABET_20261113.md`, at commit `7287e071` on `codex/minifly-consolidated-20261023`. The brief notes it was not audited by a second reader.

**Branch:** `claude/target-test-sizing-stage-1-drmnlq`. Nothing went to the default branch.

**Code:** `sz_span_clean.py`, a new file. `sz_span`, `sz_crosscut` and `sz_learner` are imported without edits.

**Records:** `runs/span_clean/`:
- `ALPHABET.json`, committed before any fit in `101e75c1`;
- `admission.json`;
- one JSON per run;
- `CLEAN_TABLES.json`;
- `logs/`.

This report gives tables, every run that lost a learned copy, and the outcomes of the four forecasts. It offers no readings beyond those. No evidence level, checker gate or review is claimed.

## Alphabet

The census covered every byte value in the first 2,000 training batches of each E26 arm-B birth stream (births 73129, 93683 and 116089), and in each birth's panels. Both were built as `exp26_candidate.dev` builds them.

All three births used the same 45 bytes: `+ - 0-9 < = > A-Z p q v w`. The panels add nothing the batches lack.

From the free printable bytes (ASCII 0x20–0x7E, space included), in ascending order:

| Role | Old | Clean |
|---|---|---|
| 24 keys | `A`–`X` | `space ! " # $ % & ' ( ) * , . / : ; ? @ [ \ ] ^ _ `` ` `` |
| 8 filler | `! # % & * + , -` | `a b c d e f g h` |
| cue | `=` | `i` |
| terminator | `\n` | `j` |

There were enough printable bytes, so no non-printable values were needed. Six of the clean keys (`! # % & * ,`) were filler bytes in the old fixture. None of them occurs in the birth's data.

## Admission

All items passed (`admission.json`):
1. With the old bytes, the generator reproduces the first roster's streams byte for byte. This was checked for fixed D = 4, 12, 24 and 40 and for mixed distances, seed 1104, 3,000 chunks.
2. With the clean bytes, every roster stream is the old stream relabelled byte for byte, with the same positions and distances.
3. Every repeat equals its key at its D, and is preceded by the cue.
4. No byte from the census occurs in any roster stream.
5. **Cost forecast:** 192,540 chunks at 0.0925 CPU-s per chunk, measured on a 300-chunk timing run (not a roster run), is 4.95 CPU-hours.

## Choices fixed before the runs

These are in the `sz_span_clean.py` docstring.

- **The stream** uses the first roster's generators and seed strings, so a clean run sees the same sequence as its old partner, relabelled.
- **"Lost a learned copy":** a window of at least 100 instances at 90% or more right, followed later by a window below 50%.
- **F8 and F10** compare repeats to 50%:
  - a run that never reaches 50% counts as needing more repeats;
  - equal counts are not fewer.

Repeats to 50% is quantized to the end of each 250-chunk window, as in the first report.

## Tables

Each cell gives **clean | old**: class, last-tenth right/total (accuracy), and repeats to 50% ("—" means 50% was never reached). Crossing figures are in `CLEAN_TABLES.json`.

### Born start, fixed D (block C)

| D | 1104 (birth 73129) | 1105 (birth 93683) | 1106 (birth 116089) |
|---|---|---|---|
| 4 | pass 5013/5013 (1.000), 21,333 \| pass 5013/5013 (1.000), 32,000 | pass 5013/5013 (1.000), 21,333 \| pass 5013/5013 (1.000), 21,333 | pass 5013/5013 (1.000), 21,333 \| pass 5013/5013 (1.000), 21,333 |
| 12 | pass 5585/5586 (1.000), 13,714 \| pass 5567/5586 (0.997), 27,428 | pass 5586/5586 (1.000), 13,714 \| pass 5571/5586 (0.997), 13,714 | pass 5586/5586 (1.000), 13,714 \| pass 5586/5586 (1.000), 13,714 |
| 24 | pass 5777/5785 (0.999), 12,307 \| pass 5275/5785 (0.912), 34,461 | pass 5774/5785 (0.998), 14,769 \| pass 5752/5785 (0.994), 12,307 | pass 5762/5785 (0.996), 17,230 \| pass 5783/5785 (1.000), 17,230 |
| 40 | **pass** 5751/5873 (0.979), 10,666 \| chance 561/5873 (0.096), — | **chance** 244/5873 (0.042), — \| partial 2590/5873 (0.441), 58,727 | chance 217/5873 (0.037), — \| chance 263/5873 (0.045), — |

### Born start, mixed D ∈ {4, 12, 24, 40} (block D)

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 4 | pass 5823/5823 (1.000), 2,228 \| pass 5797/5823 (0.996), 7,313 | pass 5709/5714 (0.999), 2,084 \| pass 5713/5714 (1.000), 2,084 | pass 5762/5762 (1.000), 2,187 \| pass 5762/5762 (1.000), 2,187 |
| 12 | pass 5881/5881 (1.000), 2,162 \| pass 5843/5881 (0.994), 6,515 | pass 5866/5868 (1.000), 2,176 \| pass 5867/5868 (1.000), 2,176 | pass 5769/5769 (1.000), 2,132 \| pass 5769/5769 (1.000), 2,132 |
| 24 | pass 5768/5768 (1.000), 2,074 \| pass 5723/5768 (0.992), 6,376 | pass 5862/5862 (1.000), 2,165 \| pass 5859/5862 (0.999), 2,165 | pass 5803/5803 (1.000), 2,193 \| pass 5801/5803 (1.000), 2,193 |
| 40 | pass 5828/5828 (1.000), 2,248 \| pass 5781/5828 (0.992), 7,377 | pass 5789/5790 (1.000), 2,208 \| pass 5782/5790 (0.999), 2,208 | pass 5850/5852 (1.000), 2,190 \| pass 5851/5852 (1.000), 2,190 |

### Random start, fixed D (block A)

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 24 | **pass** 5782/5785 (0.999), 14,769 \| partial 748/5785 (0.129), — | pass 5782/5785 (0.999), 14,769 \| pass 5784/5785 (1.000), 14,769 | **partial** 1230/5785 (0.213), — \| pass 5658/5785 (0.978), 46,769 |
| 40 | chance 246/5873 (0.042), — \| chance 239/5873 (0.041), — | pass 4717/5873 (0.803), 9,142 \| pass 4565/5873 (0.777), 50,285 | chance 224/5873 (0.038), — \| chance 237/5873 (0.040), — |

### Permuted born start, fixed D (block P)

| D | 1104 | 1105 | 1106 |
|---|---|---|---|
| 12 | pass 5586/5586 (1.000), 9,142 \| pass 5586/5586 (1.000), 9,142 | pass 5586/5586 (1.000), 9,142 \| pass 5586/5586 (1.000), 9,142 | pass 5582/5586 (0.999), 9,142 \| pass 5586/5586 (1.000), 9,142 |
| 24 | pass 5645/5785 (0.976), 19,692 \| pass 5770/5785 (0.997), 12,307 | pass 5780/5785 (0.999), 7,384 \| pass 5780/5785 (0.999), 7,384 | pass 5785/5785 (1.000), 7,384 \| pass 3253/5785 (0.562), 7,384 |

### Spans

A span is the largest D such that it and every smaller D pass in at least 2 of 3 seeds.

| Block | Clean: seeds passing by D | Clean span | Old: seeds passing by D | Old span |
|---|---|---|---|---|
| C born, fixed | 4: 3, 12: 3, 24: 3, 40: 1 | 24 | 4: 3, 12: 3, 24: 3, 40: 0 | 24 |
| D born, mixed | 3 at every D | 40 | 3 at every D | 40 |
| A random, fixed (24, 40 only) | 24: 2, 40: 1 | 24 among those tested | 24: 2, 40: 1 | 24 among those tested |
| P permuted, fixed | 12: 3, 24: 3 | 24 | 12: 3, 24: 3 | 24 |

## Runs that lost a learned copy

Each of these had a window at 90% or more, followed later by a window below 50%.

**Clean bytes:**
- **A 40 / 1105.** It was at 99–100% from window 8 to window 34. It fell to 42% in window 35, was at 54% in window 36, and at 96–99% in windows 37–39. Last tenth 0.803, so it is classed pass.
- **P 12 / 1105.** It was at 96–100% in windows 2–4. It fell to 10% in window 5 and 9% in window 6, then was at 92–100% from window 7 to window 13, the end. Last tenth 1.000, so it is classed pass.

**Old bytes**, among the runs repeated here:
- **P 24 / 1106.** A fall to 4–5% in window 23 (already reported in the first report).

The first roster's A 60 / 1106 collapse is not in this roster.

**Below the rule, for completeness:** C 40 / 1104 (clean) fell from 98% to 58% in window 12 and returned to 98% in window 13. It does not meet the rule's below-50% condition.

## Forecasts

| | Forecast | Registered | Outcome |
|---|---|---|---|
| F7 | Born start, clean bytes, D = 24: pass in 3 of 3 | about 65% | **held**: pass, pass, pass |
| F8 | Born start at D = 4 reaches 50% in fewer repeats with clean bytes than old, in all three seeds | about 60% | **did not hold** (details below) |
| F9 | Random start at D = 24 and 40 gives the same class as with old bytes in at least 5 of 6 runs | about 75% | **did not hold**: 4 of 6 |
| F10 | Born start beats permuted born in repeats to 50% at D = 12 in at least 2 of 3 seeds | about 50% | **did not hold**: 0 of 3 |

- **F8**, clean against old:
  - 1104: 21,333 against 32,000 (fewer);
  - 1105: 21,333 against 21,333 (equal, not fewer);
  - 1106: 21,333 against 21,333 (equal, not fewer).
- **F9:** the class differs at D = 24 in two runs. Seed 1104 is pass (clean) against partial (old); seed 1106 is partial (clean) against pass (old). D = 40 is the same in all three seeds. The brief says that if F9 does not hold, "the alphabet matters to a learner with no history, and the comparison above cannot be read". That is the brief's own text, recorded here, not a reading added by this report.
- **F10**, born against permuted:
  - 1104: 13,714 against 9,142;
  - 1105: 13,714 against 9,142;
  - 1106: 13,714 against 9,142.

## CPU

| Block | Runs | CPU-s |
|---|---|---|
| C | 12 | 5,062 |
| D | 3 | 4,932 |
| A | 6 | 3,828 |
| P | 6 | 2,186 |
| **Total** | **27** | **16,008 (4.45 CPU-hours)** |

- **Chunks:** 192,540, at 0.0806–0.0891 CPU-s per chunk (mean 0.0833).
- **Wall time:** 01:55:14 to 03:25:01, with three processes and one thread each, on one Linux x86_64 host (Python 3.13.18, torch 2.15.0+cpu).
- **Exit status:** all 27 runs exited 0.
- **Not included:** the census and admission, including the 300-chunk timing run.

## What was unclear

1. **Ties from window quantization.** Repeats to 50% is counted at window ends: one window is about 4,571 repeats at D = 12 and about 2,133 at D = 4. Many clean and old pairs therefore land on the same count.
   - This decides F8: two of its three seeds are ties.
   - This also decides F10: all three seeds sit one window apart.
   - The brief does not say how to treat ties. The rule here (equal is not fewer) was fixed before the runs.
2. **Seed 1104 differs from the other two in old/clean agreement.** For seeds 1105 and 1106, the clean and old born runs give equal repeats to 50% almost everywhere. Seed 1104 (birth 73129) is the seed whose figures differ in most rows: C 4, C 12, C 24, C 40 and all of D. I did not investigate this.
3. **Random-start runs at D = 24 swapped classes between alphabets.** Clean A 24 / 1106 was still rising at the end: windows 19–24 were at 7–23%. Old A 24 / 1104 was also rising: 7%, 14%, 17% in its last three windows. I extended neither stream; that is NOT_RUN.
4. **"Printable"** was taken as ASCII 0x20–0x7E, so space is the first key. A definition without space would shift every chosen byte by one position in the list.
5. **The census covers the birth's own panels only.** It covers 2,000 training batches and the birth's panels, as the brief says, and not the life panels of later E26 stages.

## Not run

- Any stream extended beyond its briefed length.
- Any investigation of the collapses, or of seed 1104's differences.
- The 84-fit sizing roster, which is still held.
- Anything from the sizing brief's revision 4.
