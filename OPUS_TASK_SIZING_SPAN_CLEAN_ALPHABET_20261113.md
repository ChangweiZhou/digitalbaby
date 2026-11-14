# Span roster, born blocks again with bytes the birth never used (13 November 2026)

Measurement; no evidence level claimed. Follows `OPUS_TASK_SIZING_SPAN_20261113.md` and its code in `TARGET_TEST_STEP0/sizing/`. Not audited by a second reader: Mr. Xu waived the audit step for this round. Same host class as the first roster; at most three processes.

## Why

The copy fixture's keys are the capital letters A–X, which the birth was trained to ignore as distractors (`exp11_cell.DISTRACT`), and its filler `!#%&*+,-` includes operator bytes of the birth. So "a born start is not a head start" may be a clash of alphabets. The outside review also notes that a permuted copy of the born weights beat the born weights in all six pairs.

## What to run

1. **Find the unused bytes.** Collect every byte value that occurs in 2,000 training batches of the E26 birth stream and in its panels, for each of the three seeds. From the printable bytes outside that set choose, in ascending order, 24 keys, then 8 filler bytes, then a cue and a terminator. If fewer than 34 are free, use non-printable values from 128 upward and say so. Write the lists and the census to `runs/span_clean/ALPHABET.json` before any fit.
2. **Fixture.** `sz_crosscut.fixture_stream` with those bytes in place of A–X, the filler, `=` and the newline; nothing else changes (dose, classes, last-tenth scoring, repeats to 50%). A new file `sz_span_clean.py`; edit nothing existing.
3. **Runs**, three seeds each:
   - born start (E26 arm B births), fixed D = 4, 12, 24, 40;
   - born start, mixed {4, 12, 24, 40};
   - random start, fixed D = 24 and 40;
   - permuted born, fixed D = 12 and 24.
   27 runs, about 5.5 CPU-hours at the first roster's pace.

## Forecasts (mine)

- **F7** Born start, clean bytes, D = 24: pass in three of three. About 65%.
- **F8** Born start at D = 4 reaches 50% in fewer repeats with clean bytes than it did with the old ones, in all three seeds. About 60%.
- **F9** Random start at D = 24 and 40 gives the same class as with the old bytes in at least five of six runs. About 75%. If it does not, the alphabet matters to a learner with no history, and the comparison above cannot be read.
- **F10** Born start beats permuted born in repeats to 50% at D = 12 in at least two of three seeds. About 50%.

## Done

`sz_span_clean.py`, `runs/span_clean/*.json`, and `SZ_SPAN_CLEAN_REPORT.md` with the same tables as the first report, old and new bytes side by side, every run that lost a learned copy named, and the four forecast outcomes. No readings beyond them.
