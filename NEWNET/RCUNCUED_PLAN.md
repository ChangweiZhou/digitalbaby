# Rule cells that also read match cells: keeping rules apart without cues (plan, written before any run, 23 November 2026)

Branch `claude/newnet-rcuncued-sei232` from `claude/magical-albattani-sei232` @ d0f0a9da. Development level, one machine.
No gate, seal or outside review is claimed. No existing .py file is changed (new files only: `net13_m.py`, alias `net131.py`).

## Question
net13's rule cells keep rules apart when each rule has its own delimiter (cue=1: t23 succ at the end 0.82; t27 map 0.74-0.88), but
without cues (cue=0) the learned mapping is lost (t27 with net12_rc = net13: map at the end 0.00-0.03). Without a cue the rule in
force is still evident in what the network reads: which earlier position the answer bytes repeat. Match cells (net12 match=K:
m[k, j] = 1 when the byte k back equals the byte k+j back) express that relation without content.

## Hypothesis (one change)
If the rule cells read the match cells as well as the lag cells (their input set x becomes lag cells + match cells, normalised as
before; the same cosine competition, recruitment, persistence, k-winner and prototype learning; the gate and every readout unchanged),
the rule cells infer the rule in force from the answers and keep the rules apart without cues.
Implementation: `net13_m.py`, `Tissue(net13.Tissue)` with option `rc_match=K` (K=4 in the runs); rc_match=0 must give net13 exactly
(checked on the 3,600-byte check of `check_net13.py` style and by the full t23 cue=1 net13 run reproducing NET13_REPORT's 777,024).
Match cells feed ONLY the rule cells (not the gate, not the direct readout).

## Runs (seed 1 first)
- t23_life_rules.py net=13 / net=131 rc_match=4, struct K8, cue=0 and cue=1 (4 runs);
- t27_newlife.py net=13 / net=131 rc_match=4, cue=0 and cue=1 (4 runs);
- OMP_NUM_THREADS=1, 4 processes at a time (nproc = 4); ~27 min a run. Logs: `rcuncued_runs/`.
- Seeds 2-3 of the same 8 runs only if the seed-1 predictions hold (all of 1-4 below).

## Predictions (seed 1; net13 cue=0 numbers are baselines measured in this round)
1. t23 cue=0: rc_match succ probe at the end >= 0.5 (net13 cue=0 baseline expected low, net12 cue=0 was 0.02).
2. t27_newlife cue=0: rc_match map probe at the end >= 0.5 (net13 = net12_rc cue=0 was 0.01).
3. cue=1 not worse than net13: t23 succ at the end >= 0.80 and bits <= 777,024 + 1% (784,800); t27 map at the end >= 0.85 and bits
   <= 812,727 + 1%.
4. Bits with cue=0 lower with rc_match than net13 cue=0, on both t23 and t27.
Falsifiers: t23 or t27 cue=0 end probe of the mapping < 0.3 with rc_match -> the hypothesis is refuted for this implementation.
Note (known limit of uncued probes): the first answer byte cannot tell the rules apart before any answer byte is read, so a rule not
in force scores at most ~2/3 if inferred from the first byte; 0.5 allows for that.

## Recorded beforehand
- Operative learner: net13 (K8 + lag cells W=8 + direct readout + relay + rule cells rc=64, rc_k=3, rc_tau=8, rc_vig=0.4,
  rc_eta=0.02) + rc_match=4 (4 x 7 = 28 match-cell slots, 22 of them can fire, as rule-cell inputs; Wr grows from 64 x 1,024 to 64 x 1,052).
- Cost: 8 runs x ~27 min on 1 core each = ~3.6 CPU-hours for seed 1; +7.2 CPU-hours for seeds 2-3 if run.
- NOT_RUN unless stated: seeds 2-3 if predictions fail; battery/natural text tests; toys.
