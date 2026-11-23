# Rule cells on a new life (23 November 2026)

Helper branch `claude/newnet-rcnewlife-sei232` from `claude/magical-albattani-sei232` @ 32db8c13. Development level, one machine.
No gate, seal or outside review is claimed. The design, runs and predictions were fixed in `RCNEWLIFE_PLAN.md` and committed before any run.
No option was retuned; no existing .py file was changed.

## Question
Are the persistent rule cells (`net12_rc.py` rc=64 rc_gate=1, defaults rc_k=3 rc_tau=8 rc_vig=0.4 rc_eta=0.02; RULECELLS_REPORT.md)
tuned to the one test they were found on (t23: copy, rev, text, succ, copy, rev with "=", "~", "+")?

## The new life (`t27_newlife.py`; the t23 measurement design)
- Lines `w D W\n`, w = 3 symbols of a 40-symbol alphabet: printable bytes except `: | # % @`, a different draw from t23's
  (15 of the 40 symbols are shared with t23's alphabet). The alphabet starts at 8 and gains one symbol every 150 lines.
- Rules: copy `w0 w1 w2`; swap `w1 w0 w2`; rot `w1 w2 w0`; skip `w0 w2 w0`; map: each symbol to its partner in a fixed random
  pairing of the 40.
- Delimiters, cue=1: copy `:`, swap `|`, rot `#`, skip `%`, map `@`. cue=0: all `:`.
- Order, 3,000 lines a block: P0 rot, P1 map, P2 text (40,000 bytes of tinyshakespeare from byte 300,000), P3 swap, P4 skip,
  P5 rot, P6 map, P7 copy, P8 swap. 29,000 line-equivalents in all.
- Scores are as in t23:
  - first-met answer bytes by layer, prequential;
  - windows of 250 lines;
  - probes with learning off every 1,000 lines (60 fresh items a rule; 1,000 never-read text bytes);
  - total bits of the life.
- Runs: base = net12 lagro=1 relay=1; rc = base + rc=64 rc_gate=1; K8 = net10 (struct K8); store = back-off table.
  - Seed 1 for all four, cue=1 and cue=0; seeds 2-3 for base and rc, cue=1 and cue=0.
  - OMP_NUM_THREADS=1, 4 processes. About 25-28 min per net12 run.
- Logs and data:
  - logs: `rcnewlife_runs/*.log`
  - results: `rcnewlife_runs/res.jsonl`
  - table: `rcnewlife_runs/table.py` -> `table.md`
  - full probe series: `rcnewlife_runs/summ.py`

## Results (all 16 runs)
- "end probe" = probes at line 29,000, the last probe.
- "map: 6000 -> 23000 -> 26000" = map probe after the first map block, after the second map block, and after the copy block.
- "first window" = right share on first-met answer bytes in the first 250 lines of the returns P5 rot, P6 map and P8 swap.

| net | cue | seed | bits total | input / answer / text | end probe copy / swap / rot / skip / **map** / text | map: after P1 (6000) -> after P6 (23000) -> 26000 | text after P2 (11000) | first-met b: P0 rot, P1 map, P3 swap, P4 skip, P6 map, P7 copy | first window P5 rot / P6 map / P8 swap | recruits | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| store | 1 | 1 | 2,002,585 | 1,137,666 / 637,764 / 227,155 | 0.04 / 0.33 / 0.58 / 0.49 / **0.07** / 0.42 | 0.05 -> 0.12 -> 0.07 | 0.42 | 0.00, 0.01, 0.01, 0.24, 0.01, 0.02 | 0.23 / 0.01 / 0.27 | - | 8 |
| K8 | 1 | 1 | 1,181,890 | 676,902 / 362,295 / 142,692 | 0.00 / 0.13 / 0.01 / 0.04 / **0.02** / 0.13 | 0.03 -> 0.15 -> 0.05 | 0.38 | 0.11, 0.07, 0.04, 0.11, 0.11, 0.10 | 0.24 / 0.06 / 0.15 | - | 1356 |
| base | 1 | 1 | 858,513 | 583,956 / 132,071 / 142,487 | 1.00 / 1.00 / 0.99 / 0.99 / **0.02** / 0.28 | 0.79 -> 0.72 -> 0.01 | 0.38 | 0.87, 0.31, 0.76, 0.74, 0.29, 0.73 | 1.00 / 0.03 / 0.76 | - | 1505 |
| base | 1 | 2 | 860,157 | 584,700 / 132,827 / 142,629 | 0.99 / 1.00 / 0.99 / 0.97 / **0.02** / 0.28 | 0.84 -> 0.68 -> 0.03 | 0.39 | 0.87, 0.31, 0.77, 0.74, 0.29, 0.72 | 1.00 / 0.02 / 0.79 | - | 1462 |
| base | 1 | 3 | 858,975 | 580,918 / 133,177 / 144,880 | 1.00 / 1.00 / 0.99 / 0.98 / **0.03** / 0.25 | 0.77 -> 0.66 -> 0.01 | 0.37 | 0.88, 0.30, 0.76, 0.73, 0.28, 0.72 | 1.00 / 0.02 / 0.75 | - | 1538 |
| rc | 1 | 1 | 812,727 | 583,376 / 86,930 / 142,421 | 1.00 / 1.00 / 1.00 / 1.00 / **0.88** / 0.29 | 0.84 -> 0.99 -> 0.93 | 0.37 | 0.96, 0.60, 0.89, 0.83, 0.83, 0.99 | 1.00 / 0.39 / 0.99 | 74117 | 1610 |
| rc | 1 | 2 | 814,973 | 584,153 / 87,984 / 142,835 | 0.99 / 1.00 / 1.00 / 1.00 / **0.79** / 0.30 | 0.95 -> 0.98 -> 0.91 | 0.39 | 0.95, 0.60, 0.89, 0.81, 0.83, 0.98 | 1.00 / 0.39 / 0.98 | 70536 | 1546 |
| rc | 1 | 3 | 815,171 | 583,566 / 86,766 / 144,839 | 1.00 / 1.00 / 1.00 / 1.00 / **0.74** / 0.27 | 0.89 -> 0.97 -> 0.92 | 0.37 | 0.96, 0.60, 0.89, 0.83, 0.83, 0.99 | 1.00 / 0.41 / 0.94 | 76642 | 1616 |
| store | 0 | 1 | 1,894,809 | 1,023,389 / 644,029 / 227,392 | 0.09 / 0.32 / 0.51 / 0.41 / **0.07** / 0.42 | 0.07 -> 0.13 -> 0.07 | 0.42 | 0.00, 0.01, 0.01, 0.23, 0.02, 0.02 | 0.33 / 0.02 / 0.25 | - | 8 |
| K8 | 0 | 1 | 1,150,937 | 656,064 / 351,476 / 143,397 | 0.07 / 0.04 / 0.07 / 0.05 / **0.02** / 0.06 | 0.06 -> 0.12 -> 0.03 | 0.35 | 0.11, 0.06, 0.04, 0.19, 0.07, 0.11 | 0.43 / 0.04 / 0.20 | - | 1364 |
| base | 0 | 1 | 870,704 | 579,955 / 147,715 / 143,034 | 0.68 / 1.00 / 0.36 / 0.04 / **0.01** / 0.25 | 0.48 -> 0.19 -> 0.00 | 0.36 | 0.87, 0.13, 0.71, 0.57, 0.11, 0.68 | 0.67 / 0.03 / 0.40 | - | 1517 |
| base | 0 | 2 | 870,164 | 579,239 / 148,098 / 142,827 | 0.68 / 0.99 / 0.37 / 0.04 / **0.00** / 0.27 | 0.51 -> 0.27 -> 0.00 | 0.38 | 0.88, 0.13, 0.71, 0.58, 0.12, 0.67 | 0.67 / 0.02 / 0.37 | - | 1497 |
| base | 0 | 3 | 869,029 | 576,876 / 147,808 / 144,345 | 0.68 / 0.99 / 0.35 / 0.03 / **0.02** / 0.24 | 0.53 -> 0.14 -> 0.01 | 0.38 | 0.88, 0.12, 0.71, 0.56, 0.11, 0.67 | 0.72 / 0.02 / 0.40 | - | 1572 |
| rc | 0 | 1 | 809,317 | 570,346 / 96,393 / 142,577 | 0.37 / 1.00 / 0.37 / 0.06 / **0.01** / 0.30 | 0.89 -> 0.94 -> 0.00 | 0.37 | 0.96, 0.49, 0.93, 0.81, 0.60, 0.95 | 0.67 / 0.03 / 0.46 | 15893 | 1602 |
| rc | 0 | 2 | 811,951 | 572,257 / 97,084 / 142,611 | 0.38 / 1.00 / 0.38 / 0.07 / **0.00** / 0.33 | 0.92 -> 0.94 -> 0.00 | 0.39 | 0.95, 0.49, 0.92, 0.81, 0.57, 0.95 | 0.68 / 0.03 / 0.46 | 19477 | 1595 |
| rc | 0 | 3 | 813,481 | 570,956 / 98,629 / 143,896 | 0.37 / 1.00 / 0.36 / 0.06 / **0.03** / 0.28 | 0.92 -> 0.95 -> 0.00 | 0.38 | 0.96, 0.48, 0.92, 0.82, 0.57, 0.94 | 0.70 / 0.02 / 0.42 | 17577 | 1666 |

The rc map probe through the last 7,000 lines (cue=1; lines 23,000 to 29,000; copy block 23-26k, swap block 26-29k):

| seed | 23k | 24k | 25k | 26k | 27k | 28k | 29k |
|---|---|---|---|---|---|---|---|
| 1 | 0.99 | 0.99 | 0.94 | 0.93 | 0.92 | 0.86 | 0.88 |
| 2 | 0.98 | 0.94 | 0.93 | 0.91 | 0.83 | 0.81 | 0.79 |
| 3 | 0.97 | 0.98 | 0.97 | 0.92 | 0.88 | 0.81 | 0.74 |

Base map probe over the same lines: 0.66-0.72 at 23k, then 0.01-0.03 at 26k and at 29k.

## Verdict against the predictions (cue=1)

| prediction | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 1. rc map at its last probe >= 0.6 | 0.88 met | 0.79 met | 0.74 met |
| 2. rc copy, swap, rot, skip >= 0.9 at the end | 1.00/1.00/1.00/1.00 met | 0.99/1.00/1.00/1.00 met | 1.00/1.00/1.00/1.00 met |
| 3. rc total bits < base | 812,727 vs 858,513 (-5.3%) met | 814,973 vs 860,157 (-5.3%) met | 815,171 vs 858,975 (-5.1%) met |

**All three predictions are met on all three seeds.** On this life the rule cells were not tuned to t23: with the t23 settings
unchanged, they keep the learned mapping where net12 loses it (0.74-0.88 against 0.02-0.03). The size of the effect is close to t23's
(succ 0.82 against 0.07; bits -6.2%). They also make the rules faster to learn:
- first-met map b: P1 0.60 against 0.31; P6 0.83 against 0.29;
- first-met swap b: 0.89 against 0.76;
- first-met copy b, met for the first time near the end: 0.98-0.99 against 0.72-0.73;
- the first window of the swap return: 0.94-0.99 against 0.75-0.79.

Limits of the result:
- **The mapping is still eroding at the end.** The map probe falls in every seed through the copy and swap blocks: 0.97-0.99 ->
  0.74-0.88 over 6,000 lines. The bar of 0.6 is met, but the decline has not stopped at line 29,000, and a longer tail would probably
  fall below it. The prediction does not show that the mapping is kept indefinitely.
- The text loss is untouched, as in t23. The text probe falls from 0.37-0.39 after the text to 0.27-0.30 at the end for rc, and to
  0.25-0.28 for base.
- The map's never-answered symbols (layer c, 13 bytes) are 0.00 for every net. This is expected: a mapping cannot be inferred for a
  symbol never given as an answer.

## cue=0 (no prediction was made)
All rules share `:`, and no net keeps the rules: at the end each net keeps mainly the rule trained last (swap 0.99-1.00).
- rc keeps the map through the second map block (0.94-0.95 at 23k), then loses it during copy: 0.00 at 26k.
- rc's other rules at the end: rot 0.36-0.38, skip 0.06-0.07, copy 0.37-0.38.
- base at the end: copy 0.68, rot 0.35-0.37, skip 0.03-0.04, map 0.00-0.02.
- Learning is still faster with rc and costs fewer bits: 809,317-813,481 against 869,029-870,704 for base (about -6.6%).
  First-met map b: P1 0.48-0.49 against 0.12-0.13; P6 0.57-0.60 against 0.11-0.12.
- This matches the expectation in RULECELLS_REPORT.md: without a cue in the stream the rule cells do not separate the rules for
  keeping.

## Other nets (seed 1)
- K8 (net10) learns none of the rules as rules: first-met b 0.04-0.11. Total bits 1,181,890 (cue=1) and 1,150,937 (cue=0).
- Storage: total bits 2,002,585 (cue=1) and 1,894,809 (cue=0). It knows only layer-a bytes (0.71-1.00).
- K8's text probe falls from 0.38 to 0.13 (cue=1).

## Limits / NOT_RUN
- One new life (one order, one alphabet draw, one pairing). K8 and storage were run on seed 1 only.
- No retuning, no longer tail, and no other battery tests.
- The rule cells still separate on a delimiter present in the stream; that is a cue, not an inferred rule (cue=0 shows the limit).
