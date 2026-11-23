# Ideas 6 and 7 on t22 (plan, written before any run; 23 November 2026)

Development level, one machine, seed 1; no gate, seal or outside review is claimed. Code: `net12_fb.py` (a subclass of
`net12.Tissue`; net12.py and the other .py files are not changed), script `t22_fb.py` (a copy of `t22_rules.py` that imports
`net12_fb`), logs and results in `idea67_runs/`. Base for all runs: struct=K8, lagro=1, relay=1, 24,000 lines, S1=12,000,
S2=18,000, GROW=250, seed 1 (the same stream as RULES_REPORT.md).

Reference numbers (rules_runs/, same stream):
- teach (res.jsonl, net12 lagro relay): P0 copy b 0.99, c 0.84; P1 rev b 0.77; rev after_change 0.34 for ~7 windows, 0.89 at window 11,
  0.95 at window 12 (~1,750 lines to switch); P2 copy b 0.91.
- yes/no, noise=0.1 (res_yesno.jsonl): P0 copy b 0.14, c 0.02; P1 rev b 0.33; P2 copy b 0.82, c 0.43.

## Idea 6: exploration under yes/no feedback (fbm=yesno)
Mechanisms (all local to the output cells; no task knowledge):
- (a) `unc=1`: the exploration noise is scaled by the network's own uncertainty, sd = unc_noise * (1 - max p), p = max(O,0)
  normalised (uniform when all outputs are <= 0). `unc=2`: sd = unc_noise * r, r a running mean (time constant 100 answers) of the
  share of answers told "wrong".
- (b) `ior=S`: inhibition of return. An answer told "wrong" adds S to an inhibition of that output line in the present context
  (key: the byte just read, i.e. the present input line); the inhibition is subtracted from the outputs when answering and decays
  with time constant ior_tau bytes (default 50). So a refuted guess is not repeated soon in the same context.
- (c) both.
Hypothesis: these raise P0 copy b under yes/no. Falsifier / success criterion: **P0 copy b >= 0.5** (full run, seed 1) for at least
one of (a), (b), (c); otherwise idea 6 fails at this level. Also reported: P0 c, P1 rev b, P2 copy b/c, after_change windows.

## Idea 7: surprise-scaled plasticity (fbm=teach)
`surp=1`: a local surprise signal from the output cells' own mismatch on each byte, err = clip(1 - O[byte], 0, 1) (before the byte is
read); a fast (tau_f = 100 bytes) and a slow (tau_s = 2,000 bytes) running mean; m = clip(fast / slow, m_lo, m_hi) (default
0.25 .. 4) multiplies every output learning rate (eta_c, holders' eta_o, eta_l, eta_r, eta_bg, eta_i, eta_h2, eta_w, eta_m) at each
step: up after a change of rule, about 1 (or below) when stable. The context cells' (net7) own unsupervised learning is not scaled.
Hypothesis: faster switch copy -> rev without loss elsewhere. Success criterion (both needed): **rev after_change reaches 0.9 within
the first 4 windows (250 lines each)** and **P0 copy b >= 0.97 and P0 copy c >= 0.75**.

Small runs (6,000 lines, S1=3,000, S2=4,500) may be used first to check the code and pick unc_noise / ior; they are reported as
such and the criteria are judged on the full runs only. Cost: one full run ~20 min CPU (single thread); 4-5 full runs in parallel.
