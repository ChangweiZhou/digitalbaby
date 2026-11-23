# IDEA4: error-driven recruitment of cells (net12_rec.py), 23 November 2026

Development level, seed 1, one machine (4 cores, OMP_NUM_THREADS=1). No gate, seal or outside review is claimed.
Source: NEWNET at claude/magical-albattani-sei232 @ 6dc127e8; this work is on claude/newnet-idea4-sei232. No existing .py was edited.
New files: `net12_rec.py` (the mechanism, subclass of net12.Tissue), `t24_add_rec.py` and `t22_rules_rec.py` (copies of the tests that
can import it; t24_add_rec also has the oracle diagnostic), `IDEA4_PLAN.md` (written before the main runs; it also lists the dev tuning),
`idea4_runs/` (logs, `small.jsonl`, `full.jsonl`, `t22.jsonl`, `posthoc.jsonl`, run scripts, dev diagnostic).

## Mechanism (in short; the docstring of net12_rec.py has all of it)
A pool of 16 candidate cells. Each has second-level synapses: a conjunction "cell and input i" for every active window input of
net12 (lag and pair cells), plus "the cell alone" (hashed, 4096 rows per cell). While a candidate, these synapses are silent: they learn
by the delta rule to fit the broadcast mismatch (e - y_c) but add nothing to the outputs. Score: running prequential error reduction
2 e.y - |y|^2 over the running |e|^2. Main arm (rec_mode=w): the cell's activity is a sigmoid of learned strengths over the window's
inputs and the earlier recruited cells; they learn by three local factors (presynaptic activity, own slope, own output's agreement with
the mismatch). Literal arms: fixed random conjunctions (2-3 lag cells, or pair + lag cell) born from the current activity, scored by the
same gain, or by plain correlation with the mismatch vector. Recruited at score >= 0.01 (corr: 0.1) after 6000 bytes (at most 8 cells,
2000 bytes apart); unrecruited at 18000 bytes: replaced. Recruited cells transmit (one more joint compartment) and feed later candidates.
Parameters were tuned on a dev run of the col stream before the main runs. The plan discloses this. They were then fixed.

## Addition (t24, fmt=col: carry not written; colc: carry written). Whole lines right, phase N1 (prequential, learning on)

| run | size | P1_old | **P1_new** | free test, 8 digits | recruited | corr(recruited h, true carry) |
|---|---|---|---|---|---|---|
| net12 lagro=1 pairs=32768, col | small | 0.436 | **0.022** | 0.00 | - | - |
| + rec w, col | small | 0.452 | **0.022** | 0.00 | 1 | -0.03 |
| + rec conj (gain), col | small | 0.438 | **0.022** | 0.00 | 0 (96 replaced) | - |
| + rec conj, corr score, col | small | 0.438 | **0.022** | 0.00 | 0 (96 replaced) | - |
| net12 lagro=1 pairs=32768, colc | small | 0.913 | **0.668** | 0.72 | - | - |
| + rec w, colc | small | 0.930 | **0.748** | 0.76 | 2 | -0.14, 0.12 |
| net12 lagro=1 pairs=32768, col | full | 0.473 | **0.035** | 0.00 | - | - |
| + rec w, col | full | 0.487 | **0.033** | 0.02 | 1 (192 replaced) | -0.10 |
| net12 lagro=1 pairs=32768, colc | full | 0.924 | **0.756** | 0.56 | - | - |
| + rec w, colc | full | 0.943 | **0.803** | 0.64 | 2 | 0.23, 0.43 |

small = struct K8 L=4 N0=N1=4000; full = K8 (L=16) N0=N1=8000. Byte level (col, full, P1 new length): s bytes with carry 0 / carry 1
right: base 0.734 / 0.512, rec 0.738 / 0.519.

Diagnostics. Neither is a mechanism. Both use small size, col:
- **Oracle carry cell.** One cell is recruited from the start, and the script sets its activity to the true carry at each s byte. Only
  its second-level synapses learn. P1_new 0.034, and carry-1 bytes at new lengths 0.52. With a 10x second-level rate (eta_rt=2):
  P1_new 0.153, carry-1 bytes 0.77, free test at 8 digits 0.10. So even a perfect carry cell, used through this second level, does not
  reach 0.30 at this size.
- **Post-hoc arm.** rec w with eta_rt=2 (this choice came from the oracle result) gave P1_new 0.021. One cell was recruited, with
  carry corr 0.10.
- **Dev run (learning only, before the main runs).** The candidates' activity reached |corr| 0.2-0.4 with the carry. It did not get
  closer, and after recruitment the corr fell to about 0.

## t22_rules (small: lines 6000, S1 3000, S2 4500, K8 L=4), right share by phase/layer: base -> + rec w (8 cells recruited)
0b 0.860 -> 0.901; 0c 0.158 -> 0.158; 0n 0.158 -> 0.237; **1a 0.882 -> 0.836 (-0.046)**; 1b 0.248 -> 0.441; 1c 0.00 -> 0.00; 1n 0.045 -> 0.045;
2a 1.00 -> 1.00; 2b 0.789 -> 0.880; 2c 0.00 -> 0.00; 2n 0.042 -> 0.125.
In the windows after a change of rule (250 lines each), the switch to rev starts slower and then gets ahead: 0.40 0.53 0.65 0.75 0.81 0.88
vs base 0.45 0.62 0.62 0.65 0.66 0.68. The return to copy is better from the start: 0.66 0.78 0.90 0.95 0.97 0.98 vs base 0.63 0.69 0.75 0.84 0.88 0.91.
Several layers have few items (1c n=6, 2a n=24, 2n n=24).

## Verdict (against the plan's criteria)
- **Main criterion failed.** Full-size fmt=col P1_new is 0.033, against 0.035 for the baseline; it needed >= 0.30. At this size and
  seed, error-driven recruitment does not learn the carry. The recruited cells do not stand for it (corr -0.10). The literal variants
  (fixed conjunctions, by gain or by correlation) recruit nothing.
- **Why, as far as the runs show.** Forming the carry fails: the candidates get partway (corr 0.2-0.4) and no further. Using it also
  fails: with a perfect carry cell, the second-level conjunctions reach only 0.03, or 0.15 at 10x rate. Both would have to improve.
  A carry alone has no correlation with any one output's mismatch, which is why the literal correlation score finds nothing.
- **colc did better, not worse.** P1_new went 0.756 -> 0.803 at full size and 0.668 -> 0.748 at small; the free test at 8 digits went
  0.56 -> 0.64 (full). This is one seed, so it is not confirmed.
- **t22 fails the plan's "no harm" rule** (no layer worse by more than 0.03). Layer 1a fell by 0.046. Five other layers rose, by 0.04 to 0.19.
- **NOT_RUN:** other seeds; the full-size t22; conj arms at full size and on colc; the hid=256 comparison. Its log in hid_runs stops at
  line 6000 and has no final result.
