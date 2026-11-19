# The output end on N0, N128 and N256 beside U (19 November 2026)

This folder is a copy of the reviewer's folder at a25450e1 on claude/vibrant-cerf-sje726 (`tissue.c`, `tis.py`, `drive.py`, `deps/`, `README.md`), plus `deps/PREP` (the scorer, copied from 10e3ab9c). Added:
- `measure.py`: runs a tissue once and does parts A, B and C below on its recorded outputs;
- `by_distance.py`: the fitted reader by distance from the input lines;
- `make_table.py` (which builds `TABLE.md`), `run_batch.sh`, `batch.txt` and `batch_dist.txt`.

Results are in `res/` and logs in `out/`. U reproduces exactly the figures of the earlier folders: set (a) 14.70, own neighbourhood 10.27, and the online rules on seed 73129.

**Tissues** (65,536 bytes; seeds 73129, 93683, 116089):
- N0: `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024`.
- N128 and N256: N0 with `tau=u2-128` and `tau=u2-256`.
- U: `rule=fixed`.

**What was measured.**
- **A. Fitted ridge readers**, as in `reach.py` and `reach_nbhd.py`. Fitted on the first three quarters and scored on the last quarter, on:
  - set (a), the cells near the output lines;
  - each byte on its own driver's neighbourhood;
  - set (b), every non-feedback cell;
  - one, two and three one-hots;
  - and the scorer's ref_1, ref_2, ref_3 and the end mark's share, all overall and at R1, R2.
- **B. Online rules** on the cells near each driver: supervised D (eta 1/64 and 1/16), yes/no A (kappa 1/8, eps 0 and 1/16) and yes/no B (eps 1/16). Each has two preparations of the inputs:
  - "run": running standardisation (lms_diag.py found fixed standardisation no better, and eta 1/64 the best step of D);
  - "raw": the outputs as they are.
- **C.** The first window of 4,096 in which each online rule reached the end mark's share, ref_1 and ref_2 of that window.

**Results** (% on the last quarter, range over the three seeds; `TABLE.md` has every cell).

| | U | N0 | N128 | N256 |
|---|---|---|---|---|
| Fitted, set (b) | 14.1–14.7 | 14.8–16.0 | 23.1–23.3 | 23.2–24.0 |
| Fitted, set (a) | 12.8–14.7 | 8.8–11.5 | 7.8–13.0 | 8.9–10.1 |
| Fitted, own neighbourhood | 8.8–10.3 | 7.1–9.7 | 7.1–8.3 | 7.2–8.8 |
| Online D, eta 1/64 (run / raw) | 4.5–6.8 / 2.8–3.2 | 4.7–9.5 / 4.6–9.1 | 4.4–7.1 / 4.5–6.9 | 4.4–7.3 / 4.8–7.2 |
| Online yes/no B (run / raw) | 2.7–5.8 / 5.8–5.9 | 4.8–8.3 / 4.1–8.2 | 4.3–4.4 / 4.1–5.6 | 2.4–4.8 / 4.3–6.8 |
| Online yes/no A, eps 1/16 (run / raw) | 0.5–3.5 / 0.3–0.6 | 0.9–6.1 / 0.4–5.5 | 1.6–2.4 / 0.4–0.5 | 1.1–3.0 / 0.3–1.8 |

References on the same positions: one-hot readers on one, two and three bytes 10.4–10.8 / 12.5–13.0 / 27.4–27.7; ref_1 10.5–10.6; ref_2 12.1–12.3; ref_3 25.1–25.4; end mark 6.2–6.3.

**C.**
- The end mark's share was reached by:
  - on N0: D and B after 8,192 bytes and A (eps 1/16, run) after 32,768 on seed 73129; D after 16,384 and B (run) after 57,344 on seed 116089; nothing on seed 93683;
  - D on N128 (seed 116089, 24,576–28,672) and on N256 (seed 93683, 24,576–28,672);
  - B on N256 after 45,056 bytes (one seed each preparation);
  - D on U once (seed 116089, 16,384).
- No online rule reached ref_1 or ref_2 in any window, on any tissue or seed.

**What the runs measured.**
1. **Fitted readers.**
   - On the whole tissue, N128 and N256 give the next byte far more than U: 23.1–24.0 against 14.1–14.7. That is above what two bytes give (12.5–13.0) and close to the table of three bytes (25.1–25.4). N0 gives a little more than U (14.8–16.0).
   - Near the output lines all three new tissues give less than U: set (a) 7.8–13.0 against 12.8–14.7; own neighbourhood 7.1–9.7 against 8.8–10.3. R1 and R2 there fall to 0.0–2.9 on N128 and N256.
2. **Online rules.**
   - On N0, seed 73129, the one-pass rules learn from the cells near the drivers. The supervised rule reaches 9.5, the fitted own-neighbourhood reader's 9.7; yes/no B reaches 8.3, above the end mark's share from 8,192 bytes on.
   - On seeds 93683 and 116089 N0 gives 4.6–5.5 to D (eta 1/64) and 4.1–5.9 to B: the learning of seed 73129 is not repeated on the other two seeds.
   - On N128 and N256 the online rules stay at 4.1–7.3.
   - No rule reaches ref_1.
3. **Raw inputs.** Raw outputs work about as well as standardised ones on the new tissues (D and B within about 1 point). On U they do not (D 2.8–3.2 raw against 4.5–6.8).

**Next byte by distance from the input lines** (`by_distance.py`; seed 73129; ridge on the cells at each distance d alone, 104–212 cells a band for d = 1 to 12; % all, last quarter; logs in `out/dist_*.log`).

| d | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N0 | 11.8 | 11.8 | 11.9 | 11.7 | 11.4 | 11.0 | 10.9 | 10.7 | 11.5 | 10.5 | 10.7 |
| N128 | 17.0 | 20.6 | 18.8 | 17.9 | 16.6 | 16.2 | 15.0 | 14.9 | 9.7 | 8.7 | 6.8 |
| N256 | 18.3 | 20.7 | 18.7 | 18.1 | 15.8 | 10.0 | 9.6 | 8.8 | 8.3 | 7.1 | 6.7 |

- Set (a)'s cells lie at d = 8 to 16 (N0) and 8 to 15 (N128, N256); most are at d = 10 to 12.
- In N128 and N256 the next byte is most readable near the input lines: 20.6–20.7 at d = 2 alone, with R2 up to 16.4 at d = 3. It falls with distance, below the present-byte level (ref_1 10.5) at d ≥ 11 (N128) and d ≥ 8 (N256). That is where the output lines' neighbourhoods are.
- In N0 it stays at 10.5–11.9 from d = 1 to 13.
- So the 23% of the whole tissue in N128 and N256 is held by cells far from the output lines.

**Not measured.**
- Lengths beyond 65,536 bytes.
- Output lines placed elsewhere than the birth's.
- The distance runs on seeds 93683 and 116089.
