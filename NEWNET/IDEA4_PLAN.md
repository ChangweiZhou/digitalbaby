# IDEA4 plan: error-driven recruitment (net12_rec.py), 23 November 2026

Development level, one machine, seed 1; no gate, seal or outside review is claimed. Source: branch claude/magical-albattani-sei232
@ 6dc127e8 (NEWNET only); work committed on claude/newnet-idea4-sei232. net12.py and every existing .py are unchanged.

## Hypothesis and falsifier
H: a pool of candidate cells that are recruited by how much their (silent) output would reduce the broadcast mismatch lets the net
form an intermediate quantity (the carry) and use it in a second level of conjunctions, so column addition without the carry
written (fmt=col) carries over to new lengths.
Falsified (at this scale, seed 1) if, at full size, fmt=col whole lines at new lengths (P1_new) stay < 0.30.

## Mechanism (net12_rec.py docstring holds the full statement)
- rec=K candidate cells; each has an activity h_c and second-level synapses T_c (conjunctions "c and input i" for every active
  window input i of net12: lag, match, pair cells; plus "c alone"), hashed into rec_P = 4096 rows per cell, V = 128 outputs each.
- While a candidate, its synapses are silent (no output) but learn by the delta rule on the broadcast mismatch e.
- Score: running mean (tau rec_tau = 2000 bytes) of the prequential error reduction 2 e.y - |y|^2 over the running mean of |e|^2.
- rec_mode=w (main arm): h_c = sigmoid(w_c . x / sqrt(n)), x = the active binary inputs and the earlier recruited cells' h (cascade);
  w learns locally: presynaptic x, own slope h(1-h), own output's agreement with the mismatch u_c.(e - y_c) (eta_rw = 5).
- rec_mode=conj (literal variant): fixed conjunctions born from the moment's activity: 2-3 (lag, byte) cells, or 1 pair + 1 lag cell.
- rec_score=corr (literal variant, with conj): plain correlation magnitude of h_c with the mismatch vector; no silent learning.
- Recruit at age >= rec_age (6000 bytes) if score >= rec_th (0.01 for gain, 0.1 for corr), at most rec_max = 8 cells, none within
  rec_gap = 2000 bytes of the last; recruited cells
  transmit (one more compartment of the joint step, eta/ncomp) and feed later candidates; unrecruited at 3 x rec_age: replaced.
- Other parameters as given: eta_rt 0.2, rec_slow 0.2, rec_init 1.0, rec=16 candidates.
Nothing is specific to addition: no digit, column or carry is named; the same parameters run on t22_rules.

Why the literal "correlation of activity with residual" is expected to be weak here (stated before running): the carry alone gives no
information on the next answer byte (s is about uniform whatever the carry), so a carry cell has ~0 correlation with each output's
mismatch; it helps only through conjunctions with the current column. The gain score measures this through the cell's own silent
second-level synapses. Both are run.

## Runs (OMP_NUM_THREADS=1, <= 4 processes)
Small (struct=K8 L=4 N0=4000 N1=4000, seed 1), fmt col and colc:
 1. baseline: net=12 lagro=1 pairs=32768 (t24_add.py)
 2. rec w:    net=12rec lagro=1 pairs=32768 rec=16 (t24_add_rec.py)
 3. rec conj (gain), 4. rec conj corr (rec_th=0.1) - col only unless they recruit.
Full (L=16 default, N0=N1=8000, seed 1): the best rec arm on col and colc; baseline at full size is the existing hid_runs/add_runs
numbers (net12 lagro=1 relay=1: col P1_new 0.031, colc 0.739; lagro=1 pairs=32768 full run not finished in hid_runs).
t22_rules small run (lines=6000 S1=3000 S2=4500, struct K8 L=4): net12 lagro=1 pairs=32768 vs the same + rec arm.
Diagnostic (not used by the net): corr(recruited h, true carry) on s bytes of phase N1.

## Success criteria (fixed before the main runs)
- Success: full size fmt=col P1_new >= 0.30, and fmt=colc P1_new not below the baseline's (net12 0.739) by more than 0.02.
- Small size: report P1_new col/colc against the small baseline; a small-size pass alone is not success.
- t22: no layer's right share (phases 0-2, layers a/b/n/c) worse than the baseline by more than 0.03 = "no harm".
- Honest NOT_RUN: anything not run is listed as such.

## Cost
Small runs ~ 15-25 min each (4 parallel); full runs ~ 1-2 h each; CPU only, ~ 4 cores.

## Development before the main runs (disclosed: parameters were tuned on the addition stream itself)
idea4_runs/dev_scores.py, L=4, 1500-2500 lines of the col stream, generator seed 5 (not the t24 stream seed), learning only.
- First version: silent synapses learned on e (not e - y_c): they never converge; scores -0.06..-0.10. Fixed (fit e - y_c).
- eta_rw 0.05 -> |corr(h, carry)| ~0.1, scores < 0; eta_rw 1 / 5 / 20 (eta_rt 0.05) and 5 (eta_rt 0.2): at 2500 lines
  max |corr| 0.40 / 0.37 / 0.26 / 0.29, best score 0.004 / 0.011 / 0.016 / 0.018. Chosen: eta_rw 5, eta_rt 0.2, rec_th 0.01,
  rec_age 6000; the refractory rec_gap was added because all candidates learn about the same thing.
So the carry is only partly formed by the candidates (corr 0.2-0.4, a clean carry cell would be ~1) before any recruitment.
The parameters are then fixed for every run below (addition and t22).
