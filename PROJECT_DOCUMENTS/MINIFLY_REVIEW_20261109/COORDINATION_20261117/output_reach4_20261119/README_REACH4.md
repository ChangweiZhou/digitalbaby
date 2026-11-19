# The output end on the long-axon tissues (19 November 2026)

This folder is a copy of the reviewer's folder at 7eebe2a2 (`tis.py` taken again at 39740a9f for `longfast`), built with `-ffp-contract=off`, plus `deps/PREP` (the scorer). Added:
- `measure.py` (as output_reach3_20261119, with a third online form, "clip", added at 11:20 UTC for every later run);
- `measure_b.py` (the online part only, for long runs);
- `diag_std.py` (the check of the standardisation);
- `by_class.py` (fitted readers by class of the predicted position);
- `make_table.py` (builds `TABLE.md`), `run_batch.sh`, `slot_run.sh`, `slot_mixed.sh`, and the batch files.

Logs are in `out/`, results in `res/`.

**Reading end:** the birth's drivers at x = 1, each seeing its own neighbourhood. The neighbourhood is enlarged by the long-axon cells: a driver has a median of 121 inputs on L8 and 152 on A4, against 93 without long axons. The drivers only read; the tissue is not changed by them.

**Tissues** (65,536 bytes; seeds 73129, 93683, 116089):
- L8-0: `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2`.
- L8-64, L8-128, L8-256: with `tau=u2-64`, `u2-128`, `u2-256`.
- LF-256: L8-256 with `longfast=1`, seed 73129 only; the other seeds were withdrawn by the reviewer.
- A4-0, A4-128, A4-256: `long=1-1/4` in place of `long=8-1/2`.

**Online forms:** run (running standardisation), clip (the same clipped at ±8; only in runs started after 11:20: L8-128, LF-256, A4), raw.

## Results (% on the last quarter; seeds 93683 / 73129 / 116089)

| Tissue | Fitted set (a) | Fitted own | D (run) | B, each form per seed: run; clip; raw | Yes/no reaches ref_1 (first window) | Yes/no reaches ref_2 |
|---|---|---|---|---|---|---|
| L8-256 | 11.7 / 18.3 / 19.1 | 10.5 / 17.2 / 17.4 | 9.3 / 17.0 / 16.2 | 4.7 / 12.0 / 9.9; no clip; 6.2 / 8.4 / 6.6 | 73129 (8,192), 116089 (53,248) | 73129 (8,192; held to 262,144) |
| L8-128 | 14.8 / 18.3 / 16.8 | 13.4 / 16.0 / 14.7 | 12.0 / 15.5 / 13.9 | 7.0 / 10.6 / 9.3; 7.9 / 13.0 / 9.1; 7.4 / 8.6 / 7.2 | 73129 (12,288) | 73129 (clip, 16,384) |
| L8-64 | 13.5 / 15.6 / 16.4 | 11.0 / 14.1 / 13.2 | 10.2 / 13.4 / 12.4 | 6.5 / 10.7 / 8.6; no clip; 5.2 / 7.8 / 6.6 | 73129 (16,384) | none |
| L8-0 | 7.7 / 10.1 / 9.7 | 7.1 / 9.7 / 8.3 | 5.2 / 5.5 / 6.5 | 5.0 / 7.8 / 4.8; no clip; 4.9 / 7.7 / 5.8 | none | none |
| A4-256 | 15.0 / 17.5 / 17.7 | 11.2 / 14.4 / 15.5 | 10.6 / 13.8 / 15.3 | 6.9 / 10.8 / 9.3; 5.6 / 10.5 / 10.0; 3.6 / 7.0 / 9.0 | 73129 (16,384), 116089 (24,576) | none |
| A4-128 | 16.6 / 16.8 / 16.6 | 13.7 / 12.8 / 14.6 | 13.8 / 11.8 / 13.8 | 7.5 / 10.2 / 10.0; 7.1 / 9.5 / 9.7; 8.2 / 7.5 / 6.6 | 73129 (16,384), 116089 (61,440) | none |
| A4-0 | 10.5 / 10.0 / 11.4 | 9.5 / 9.1 / 10.6 | 7.1 / 6.0 / 8.6 | 8.3 / 4.5 / 5.6; 5.7 / 6.9 / 5.4; 6.4 / 6.9 / 6.2 | none | none |

References on the last quarter: ref_1 10.5–10.6, ref_2 12.1–12.3, end mark 6.2–6.3; one-hot readers on one, two and three bytes 10.4–10.8 / 12.5–13.0 / 27.4–27.7. The full table, with every rule and form, is `TABLE.md`.

**By class** (`by_class.py`, fitted on set (a), last quarter; TASK / SIGN / EQ / END / R1 / R2):
- L8-256: 73129 23 / 24 / 96 / 100 / 13 / 13; 93683 0 / 61 / 41 / 72 / 0 / 0.
- A4-256: 73129 16 / 20 / 94 / 99 / 16 / 12; 93683 0 / 51 / 73 / 79 / 10 / 4; 116089 11 / 49 / 89 / 93 / 14 / 15.
- Two one-hots: 25 / 0 / 52–55 / 48 / 10–12 / 9–11.
- The tool's inputs (outputs below 0.01 set to 0) give the same as the standardised outputs throughout.

**The check of the standardisation** (`diag_std.py`, L8-256 seeds 93683 and 73129, L8-64 seed 93683):
- Standardised inputs reach |z| = 30–55, from rare excursions of cells with ordinary deviations.
- Flooring the deviation at 1/64 of the largest changes neither the inputs' extremes nor the fitted readers.
- Clipping at ±8 or flooring changes yes/no B by −0.3 to +3.4 points; it lifts B on L8-256 seed 93683 from 4.1 to 7.3–7.6, below ref_1.
- The fitted readers give R1 and R2 in-sample as they do out of sample (seed 93683: 1.0 / 0.9 against 0.5 / 0.0), so the fit does not break on new data.

**Long run** (`measure_b.py`, L8-256 seed 73129, 262,144 bytes, run form): B 11.8–12.6 per 32,768 bytes against ref_2 11.7–12.5; A rises to 12.3; D 16.2–17.1.

**What the runs measured.**
1. **Long axons with time constants** (L8-64 to L8-256, A4-128, A4-256) give a fitted reader near the output lines 11.7–19.1, against 9.7–14.7 for N0 to N256 (output_reach3), and the supervised one-pass rule 9.3–17.0.
2. **Yes/no alone, at the output lines as defined:**
   - reaches ref_1 on L8-256 on two of three seeds, and on A4-128 and A4-256 on two of three;
   - reaches ref_2 only on seed 73129 (L8-256, L8-128 with clip, LF-256 with clip), and holds it there over 262,144 bytes;
   - on seed 93683 no tissue gives yes/no ref_1.
3. **A4** evens out the fitted reader across seeds (set (a) 15.0–17.7 on A4-256) and restores "=" and R1 on seed 93683. It does not lift yes/no above L8-256: the best last quarter is 10.8, against 12.0.

**Not measured.**
- A4 and L8 beyond 65,536 bytes (except the one long run).
- LF-256 on seeds 93683 and 116089 (withdrawn).

## Rule Q (`measure_q.py`, the reviewer's message of 13:05 UTC; `TABLE_Q.md`)

**The rule.**
- A unit's output a_v = W[v] U is its own estimate of the chance that its byte comes next. Weights start at 0.
- Emission as in B: the largest output, or with chance 1/16 a value seen so far, at random.
- The emitter, on yes and on no: W[e] += eta (ok − a_e) U_e / |U_e|^2.
- On yes, every other unit: W[v] += eta (0 − a_v) U_v / |U_v|^2. On no the others do nothing.
- Qe is the emitter's change only.
- Choices taken: n = |U_v|^2 of the unit's own inputs and the constant (as D); the random emitter is treated as any emitter; outputs are not clipped to [0, 1].
- Inputs in the forms run and clip. B (eps 1/16) and D (eta 1/64) are run beside Q in the same program. B draws from a random-number generator shared with the other rules, so it differs slightly from earlier runs.

**Seeds.** 93683, 73129 and 116089 on L8-256, L8-128, A4-256 and A4-128. 146821, 224268 and 746958 were added at the reviewer's request, on L8-128 and A4-256 only. The fitted readers and `by_class.py` were run for those three seeds.

**Results** (% on the last quarter, run form, Q at eta 1/16 unless stated):

| Tissue | D (1/64), per seed | Q, per seed | Seeds where some form of Q (eta 1/16 or 1/64) reaches ref_2 in a window |
|---|---|---|---|
| L8-256 (93683 / 73129 / 116089) | 9.3 / 17.0 / 16.2 | 8.0 / 14.8 / 14.2 | 2 of 3 (73129, 116089) |
| L8-128 (93683 / 73129 / 116089 / 146821 / 224268 / 746958) | 12.0 / 15.5 / 13.9 / 9.3 / 9.6 / 9.7 | 9.9 / 13.4 / 11.8 / 8.4 / 8.7 / 8.3 | 2 of 6 (73129, 116089) |
| A4-256 (same six seeds) | 10.6 / 13.8 / 15.3 / 12.5 / 18.7 / 10.1 | 9.7 / 12.2 / 13.5 / 11.2 / 15.8 / 9.4 | 3 of 6 (73129, 116089, 224268) |
| A4-128 (93683 / 73129 / 116089) | 13.8 / 11.8 / 13.8 | 12.4 / 11.4 / 12.4 | 3 of 3 |

- Q (eta 1/16) is above B on every tissue and seed: by 0.6 to 5.5 points, and 0.8 to 0.95 of D where D is above ref_2.
- Qe equals Q at eta 1/16 and 1/64, and is worse at 1/4.

**Long run, L8-256 seed 73129, 262,144 bytes.**
- Q at eta 1/64 holds 15.0–15.6 per 32,768 bytes (last quarter 15.5; D 17.1; ref_2 11.7–12.5).
- The largest row norm of W from 65,536 bytes to the end: Q 0.386 → 0.428 at eta 1/64, 0.407 → 0.592 at 1/16, 0.591 → 1.21 at 1/4; D 0.373 → 0.439; B 13.2 → 51.8.

**By class on the further seeds.** On L8-128 near the output lines:
- TASK after the end mark 0–2 (the present byte alone gives 24–26);
- "=" 46–71 and the end mark 58–65;
- the answer digits 6–9.
On A4-256: TASK 23–26, "=" 47–96, the end mark 78–100, the answer digits 1–13.

## Removals from the addition system (`addcheck3.py`, the reviewer's message of 15:40 UTC; `TABLE_ADD3.md`)

What `addcheck2.py` variant carry still lays down, removed one at a time (the options are described at the head of `addcheck3.py`):
(a) the carry cell replaced by learned strengths (reward-modulated perturbation); (b) the four chosen holding cells replaced by a pool
of 39 holding cells and adding cells over pairs of them; (c) the two strengths of each adding cell made unequal (1 and 0.9), without
and with a local equalising rule; (2) the single answering cell replaced by the output part of M = 8 cells. D = 100, 240,000 lines,
5 orders a seed, seeds 73129 and 93683. Comparison: `addcheck2.py` variant carry at 240,000 lines, never shown whole 83.7 / 88.2 / 90.5.

Results (never shown whole, smallest / median / largest over 5 orders; full rows in `TABLE_ADD3.md`):

- (2) M = 8: 87.9 / 89.5 / 92.1 (seed 73129) and 89.6 / 90.0 / 92.0 (93683). Shown 90.2 and 90.9 (medians).
- (c) strength 0.1 without the rule: 28.4 / 31.5 / 31.8 and 27.6 / 29.9 / 32.8.
- (c) strength 0.1 with the rule: 25.2 / 27.0 / 29.9 and 27.3 / 27.5 / 30.9 at 240,000 lines. The medians were 40.9 and 39.7 at 60,000
  lines and 70.1 and 68.5 at 120,000, then fell. The strength ratios at the end were 0.94–1.02 (they were 0.9 without the rule). Why
  it falls after 120,000 lines was not examined.
- (b) pool, all 741 pairs: 6.5 / 7.1 / 8.5 and 6.2 / 6.8 / 7.1. Never shown tens 32–33, units 21–22.
- (b) pool with 2,000 carry adding cells added: 5.3 / 6.8 / 9.4 and 5.7 / 6.2 / 7.6. No better than the pool without them.
- (b) pool at D = 10, 60,000 lines: shown 85.0 and 87.5 (medians), never shown 0.0 on both seeds (largest 5.0).
- (a) learned carry and base: NOT_RUN at 240,000 lines. The batch file `batch_add3.txt` had a trailing space on the base lines, so
  `xargs -L1` joined each base line to the next learned line. Base stopped with a ValueError on its arguments
  (`out/add3_base_*.log`), and learned never started.
- Seed 116089: NOT_RUN for every variant (not started before the stop instruction of 16:13 UTC).

Earlier trials (one order, seed 73129, 60,000 lines; reported by message, no result file kept): base 42.3, learned carry 17.8
(never shown tens 35.1), strength 0.1 without the rule 18.0, M = 8 41.3. These are single trials, not the table's runs.

Limits: two seeds of three; one program and one worker; no one else has run or read `addcheck3.py`.
