# The output end placed at x0 (19 November 2026)

**Question** (the reviewer's message of 09:04 UTC): is there a place x0 and a tissue at which an output end told only yes or no passes ref_1 (the present byte's level), and one at which it passes ref_2?

**Programs.**
- `measure_x0.py` runs a tissue once, 65,536 bytes, then measures at x0 = 3/8, 1/2, 5/8, 3/4 and 1.
- `measure_x0b.py` is the same with the places given (`x0s=`) and an option `online_only=1` that skips the fitted readers. It is used for the long runs.
- `make_table_x0.py` builds `TABLE_X0.md`. Batches: `batch_x0.txt` and `batch_x0_long.txt`; logs in `out/batch_x0*.log`; results in `res/x0_*.json`.

**Reading end at x0.**
- 256 points (x0, y_v), with y_v drawn by `numpy.random.default_rng(seed * 1000 + round(1000 x0))`.
- For each point in turn, the driver is the nearest non-feedback cell not yet taken.
- The driver's inputs are the non-feedback cells within 1/8 of the point: median 86 to 97 at x0 < 1, and 43 at x0 = 1.
- The tissue runs as it is; only the place of the reading end changes. At x0 = 1 this construction stands in for the birth's drivers, with other y.

**Tissues.**
- N0: `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024`.
- N64 and N256: N0 with `tau=u2-64` and `tau=u2-256`.
- Seeds 73129, 93683 and 116089.

**Measures.**
- **A. Fitted ridge readers** (as `measure.py`): on all non-feedback cells of the band |x − x0| ≤ 1/8, and on each driver's own inputs. The one-hot readers and the scorer's references are computed once for each seed.
- **B. Online rules on each driver's own inputs**, taken as they are, with a constant 1: supervised D (eta 1/64), yes/no B (eps 1/16), and yes/no A (kappa 1/8, eps 0 and 1/16).
- **C.** For each rule, the first window of 4,096 bytes whose share reached the end mark's share, ref_1 and ref_2 of that window.

**Results:** `TABLE_X0.md`, and the section below.

## Results at 65,536 bytes (three seeds; `TABLE_X0.md`)

References on the last quarter: ref_1 10.5–10.6, ref_2 12.1–12.3, ref_3 25.1–25.4, end mark 6.2–6.3. One-hot readers on one, two and three bytes: 10.4–10.8 / 12.5–13.0 / 27.4–27.7.

1. **Fitted readers.**
   - On N64 and N256 the band halfway across holds the next byte far better than the band at x0 = 1. N64: 19.9–20.3 at x0 = 3/8, against 7.5–16.2 at 1. N256: 17.6–21.5 against 7.1–9.6.
   - On each driver's own inputs (about 90 cells) the fitted reader gets 12.7–16.1 on N64 at 3/8 to 1/2.
   - N0 is flat across x0 (band 8.1–12.7).
2. **Supervised one-pass rule D.** On N64 at x0 = 3/8 and 1/2 it gets 11.5–15.9 on all three seeds, with ref_1 reached at 8,192 bytes and ref_2 at 4,096–8,192. On N256 at 3/8 it gets 9.7–17.5.
3. **Yes/no alone.**
   - ref_1 was reached only on N64, on two of the three seeds:
     - seed 93683 at x0 = 3/8: B from 16,384, A (eps 0) from 32,768, A (eps 1/16) from 36,864; last quarter 10.7 for all three;
     - seed 73129 at x0 = 1/2: B from 28,672, last quarter 10.6; A (eps 1/16) at 65,536. At 5/8 also, B from 32,768 and A from 65,536.
   - ref_2 was reached once: seed 93683, x0 = 3/8, B, window 12,288–16,384, 11.9 against 11.5. It was not held afterwards (B 10.5–11.9).
   - On seed 116089 the yes/no rules on N64 stay below ref_1 but are still rising at 65,536 (B 3.5 → 9.5 at 3/8).
   - On N256 and N0 no yes/no rule reaches ref_1 at any place. Their best are B 9.9 (N256, 3/8) and B 7.3 (N0).
   - At x0 = 3/4 and 1 no yes/no rule reaches ref_1 on any tissue.

**Answer at 65,536 bytes.**
- **ref_1:** yes. On N64 with the output end at x0 = 3/8 to 5/8, an output end told only yes or no passes ref_1, on two of three seeds.
- **ref_2:** in a single window on one seed only, and it does not hold.
- The long runs below continue the three N64 cases to 262,144 bytes.

**Process note.** At 10:13 UTC the three long runs were started while three batch runs were still going: six processes for a few seconds. They were stopped at once and started again when one batch run was left.
