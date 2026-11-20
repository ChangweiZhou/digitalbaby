# Line `nano/layered` (track M): holding cells that read only lines, and no crossing (20 November 2026)

By the agent session of this line. Development level; five reported seeds where a median is given; repeated by no one; no checker gate, seal or review is claimed. Every figure is in a plan with its outcome (`PLAN_1.md` to `PLAN_5.md`), in `report.txt` and in `res/`. `hold.py`'s figures are quoted from `loop_small_case/REPORT.md` and `PLAN_11.md` at 0ba9f241 and were not rerun; the texts, lines and scoring here are copies of `hold.py`'s.

**The system** (`layered.py`, the reference; `kernl.c` with `fastl.py`, compiled; `check_fastl.py`: 38 cases agree to 3e-13, traces agree with difference quotients). Cells A hold: each reads only lines, so each strength's forward trace is its whole credit. Cells B, logistic, read all cells A at the same byte and the present byte's lines. A least-squares answering cell reads the cells B. The mismatch goes to the cells B over return strengths they hold, and from them, as one number from each cell B, to the cells A over return strengths the cells A hold, which learn by the same product as the forward strengths. No value differs from one cell read to the next; nothing passes backward over the text. Numbers that learn are within 1% of `hold.py`'s on each text.

| Text (measure) | layered, no crossing | cells A never change | `hold.py`, one crossing (quoted) | `hold.py`, none (quoted) |
|---|---|---|---|---|
| `g1`, one digit across 0 to 8 letters (sums shown), 960,000 lines | **96.8** (92.6 to 99.0) | 10.3 | 98.6 | 44.4 |
| `m1`, the same mod 10 (sums shown) | **51.4** (9.3 to 99.6); ten further seeds: 81.6 (6.0 to 94.3) | 10.1 | 91.2 | 27.6 |
| `d2`, two-digit sums, two delay lines (never-shown sums), 1,600,000 lines | **86.3** (74.5 to 92.1) | 0.8 | 65.8 | 23.4 |
| `g2`, two-digit numbers, 0 to 8 letters between (never-shown sums at 8 letters), 2,400,000 lines | 8.3 on seed 1; **3.7** on five seeds; 51 on seed 1 at 9,600,000 lines | 0.6 | 8 (seed 1) | 3 (seed 1) |
| `g2`, with a gate on what a cell A gives out, equal numbers | **64.7** (41.7 to 79.5); seed 1: 84.8 | | | |

**Answer to the line's question** (my reading). On three of the four texts the crossing is not needed: cells of two kinds, with exact traces and a relay over two steps, reach or pass what `hold.py` reached with the crossing (`g1` level; `d2` 86 against 66; `g2` 8 against 8 at equal length, and 65 with the output gate). On the fourth, `m1`, the layered system depends on the birth: of 16 births 9 end above 85%, 6 at or below 40%; `hold.py` with the crossing had 78 to 100 on five seeds.

**Seen besides.** (i) Sums never shown on the single-digit texts (20 pairs, no letters; small sample): the layered system answers about half of them where it learned the text (`g1` 38 to 63%; `m1` 32 to 62% on six further seeds, 77 and 85% on two of the first five), `hold.py` with the crossing 0 to 12% in its own result files. (ii) The output gate (crosswalk item 5) was the largest single effect of the line: 3.7 to 64.7 at equal numbers and length; a one-gate control at the gate arm's split stays at 3.9. The one-gate arm does rise later (51% at 9,600,000 lines on seed 1), so what is shown is a rise brought forward by a factor of about five and a higher level at 2,400,000 lines, not that one gate cannot.

**Predictions:** 34 registered; failed: L1, L5, G1, G2, G4, O4, S1, S4, S5; G3 held only in an empty sense; G7 not run. Three that held had chances of 0.15 to 0.30 from me (O1 to O3).

**Not run:** `hold.py` (quoted throughout); a comparator by back-propagation through time; K cells A drawn at birth for each cell B (all are read); the output gate on `m1`, `g1`, `d2`, in `hold.py`'s own cell, beyond 2,400,000 lines, or born open on the reported seeds; the one-gate long run on the reported seeds; more lines for the `m1` seeds that did not learn; never-shown sums at gaps of 4 and 8 on the single-digit texts; any look inside the cells; keeping; language. Cost of the line: about 9 CPU-hours.

    python3 fastl.py; python3 check_fastl.py
    python3 layered.py <label> text=m1|g1 gap=8 seed=<s> etaA=1/4 etaB=1 nB=32 lines=960000 every=80000 fast=1
    python3 layered.py <label> text=d2 chain=2 NA=160 NB=106 seed=<s> etaA=1 etaB=1 nB=32 lines=1600000 every=200000 fast=1
    python3 layered.py <label> text=g2 gap=8 chain=2 NA=160 NB=118 seed=<s> etaA=1 etaB=1 nB=32 lines=2400000 every=200000 fast=1
    python3 layered.py <label> text=g2 gap=8 chain=2 og=1 ob=-1 NA=120 NB=134 seed=<s> etaA=1/4 etaB=1 nB=32 lines=2400000 every=200000 fast=1
    python3 report.py > report.txt
