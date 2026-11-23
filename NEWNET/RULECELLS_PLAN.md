# Rule cells and attention-gated plasticity of the relay gate: plan (23 November 2026, written before the runs)

Helper branch `claude/newnet-rulecells-sei232` (from `claude/magical-albattani-sei232` @ 9dcc213e). Development level, one machine,
seed 1 only; no gate, seal or outside review is claimed. New files only: `net12_rc.py` (subclass of net12.Tissue),
`toys/vig_toy_rc.py` (a copy of toys/vig_toy_lesion.py importing net12_rc), and, only if a toy variant passes, `t23_life_rules_rc.py`.

## Problem
Life copy 2000 -> rev 2000 -> succ 3000 -> copy 1000 lines (cued "=", "~", "+"). The succ mapping is kept in the direct readout,
but copy training reopens the relay's gate through the symbol lag cells that both rules activate (succ probe after the last copy
0.58; with match=4 0.81; relay lesioned 0.92-0.98).

## Hypotheses (one each, with falsifier)
A. Persistent rule cells (a prefrontal task set). NR cells; each has a prototype over the gate's inputs (lag and match cells). At each
   byte the cell whose prototype best matches the present inputs (cosine) ignites (activity 1); if even the best match is below a
   vigilance rc_vig, the least-used cell is recruited (prototype = present inputs). An ignited cell stays active by
   self-excitation, decaying with time constant rc_tau bytes (about one line), and only the rc_k most active cells stay on
   (k-winner competition). The ignited cell's prototype moves toward the inputs (Hebbian with competition: only the winner learns;
   rate 1/uses, floored at rc_eta). Optional: a strong surprise (error on the byte read above its running mean by rc_sur) silences
   all rule cells. The relay gate reads the rule cells' activity (normalised) through its own strengths Gr (delta rule, as net12's
   gate): rc_gate=1 with the lag cells (added), rc_gate=2 instead of them. Nothing refers to any byte.
   Prediction: "+" lines ignite cells copy lines never ignite, so copy cannot reopen the gate there.
   Falsifier: succ after the last copy not above the matching baseline (0.58 / 0.81 with match=4) by >= 0.1.
B. Attention-gated plasticity of the gate (after Roelfsema's AGREL, in a local form). Each gate synapse (lag j <- input i) keeps,
   over the times input i is active, a running mean m and mean square v of the gate's error on lag j's line (time constant ag_tau
   activations). Its plasticity is scaled by the predictiveness p = (|m| + c) / (sqrt(v) + c) (c = 0.01; 1 for an input that has
   seen no error), raised to ag_pow: inputs whose correlation with the error has changed sign (symbols, used by several rules) stop
   being rewritten; consistently predictive ones (a delimiter at a lag) keep learning. This weights by predictiveness, not by
   firing rate (gate_rate failed).
   Falsifier: as for A.

## Criterion (toy, decided now)
Pass: succ probe after the last copy >= 0.85 AND copy and rev probes after the last copy >= 0.95 AND first-200 copy (first phase)
>= 0.6. Runs: A (rc_gate 1 and 2), B, A+B, each without and with match=4 (8 runs, plus a few settings of rc_k / rc_tau / ag_tau
reported as such, all of them, not only the best). Baselines: vig_toy_lesion as is (0.58) and match=4 (0.81).
If any passes: full size, `t23_life_rules_rc.py` (a copy of t23_life_rules.py importing net12_rc), net=12 K8 lagro=1 relay=1 cue=1
seed 1 with the passing settings; compare with LIFE_RULES_REPORT.md (succ 0.98 -> 0.07, text 0.35 -> 0.24, bits 828,527).
Full-size success: succ at end >= 0.5, copy/rev at end >= 0.95, bits within 2% of 828,527 (as in the idea12 plan).
If none passes: full size NOT_RUN.
