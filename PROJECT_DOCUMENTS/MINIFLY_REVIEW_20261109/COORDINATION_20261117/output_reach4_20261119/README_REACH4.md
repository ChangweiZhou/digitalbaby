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

| Tissue | Fitted set (a) | Fitted own | D (run) | B run / clip / raw | Yes/no reaches ref_1 (first window) | Yes/no reaches ref_2 |
|---|---|---|---|---|---|---|
| L8-256 | 11.7 / 18.3 / 19.1 | 10.5 / 17.2 / 17.4 | 9.3 / 17.0 / 16.2 | 4.7 / 12.0 / 9.9 (run) | 73129 (8,192), 116089 (53,248) | 73129 (8,192; held to 262,144) |
| L8-128 | 14.8 / 18.3 / 16.8 | 13.4 / 16.0 / 14.7 | 12.0 / 15.5 / 13.9 | 7.0–7.9 / 8.6–13.0 / 7.2–9.3 | 73129 (12,288) | 73129 (clip, 16,384) |
| L8-64 | 13.5 / 15.6 / 16.4 | 11.0 / 14.1 / 13.2 | 10.2 / 13.4 / 12.4 | 4.4–6.5 / 7.8–10.7 / 6.6–8.6 | 73129 (16,384) | none |
| L8-0 | 7.7 / 10.1 / 9.7 | 7.1 / 9.7 / 8.3 | 5.2 / 5.5 / 6.5 | 4.8–5.8 / 7.7–7.8 / 4.8–5.8 | none | none |
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
