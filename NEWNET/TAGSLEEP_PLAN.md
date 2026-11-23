# Tag-and-capture (idea C) and sleep downscaling (idea D): plan, written before any run (23 November 2026)

Helper branch `claude/newnet-tagsleep-sei232`, from `claude/magical-albattani-sei232` @ 9dcc213e. Development level, one machine,
seed 1; no gate, seal or outside review is claimed. Existing .py files are not edited: new `net12_ts.py` (subclass of
`net12_rep.Tissue`), `toys/ts_toy.py`, and (only if a toy passes) a new full-size script `t23_life_ts.py`.

## Problem (from NOTES_20261123.md, IDEA12_REPORT.md)
Life copy -> rev -> succ -> copy (cued). The succ mapping survives in the direct readout, but copy training reopens the relay gate
through the symbol lag cells both rules share; succ probe after the last copy 0.58 (no replay), 0.72 (sleep: 1,000 stored lines
replayed after each phase, pairs=32768). Reference that already passes the toy: match=4 + targeted replay 1 per 4: 0.93.

## Hypothesis and falsifier
H: if lasting change of the plastic strengths (relay gate, optionally the direct readout) requires a novelty signal while a decaying
tag is set (C), and/or sleep scales recent small strengths away (D), then the small steady drift of the gate during the returning
(familiar) copy rule fades instead of accumulating, and succ is kept after the last copy.
Falsified (on the toy) if no variant of C, D, C+D (with or without match=4) reaches the criterion below; and C in particular is
falsified if the novelty signal is high during the copy return (then copy's changes are captured like any other).

## Criterion (toy, 4 regions, seed 1, probes with learning off; fixed before the runs)
- succ probe after the last copy phase (incl. its sleep if any) >= 0.85;
- copy and rev probes at that point >= 0.95;
- first-200 copy lines right share (first phase) >= 0.6.
All three must hold. Partial: succ gain >= +0.10 over the same configuration without C/D.

## Mechanisms (general, local; not tuned to the test)
C (tag=1): each plastic strength is S + T (S: lasting; T: tag). net12's local rules change the sum; the change goes into T. Every
byte read with learning on: T decays (time constant tag_tau bytes, default 300) and a share cap * nov of T is captured into S.
Novelty nov in [0, 1] is a local scalar (a dopamine-like broadcast):
  nov=1  error: e = clip(1 - O[byte read], 0, 1) (O before the byte); fast mean (nf=50 bytes) and slow mean (ns=5,000 bytes);
         nov = clip((fast - slow) / nsc, 0, 1), nsc = 0.2.
  nov=2  input: d = 1 - cos(fast byte trace, slow byte trace) (taus as above); its own slow mean; nov = clip((d - mean d)/nsc, 0, 1).
  nov=0  constant nov = nconst (control: a decaying tag with uniform capture).
tagB=1 applies the same to the direct readout B. Defaults cap = 0.02 per byte, tag_tau = 300.
D (sleep downscaling, called by the script at the start of each sleep, before the offline replay): ds_frac: G *= 1 - ds_frac;
ds_sub: G = sign(G) max(|G| - ds_sub * mean|G|, 0) (relative to the mean absolute strength, so it is scale-free); ds_all=1 applies
to all compartments' strengths (B, net10's context and holder readouts too). With tag=1 the downscaling acts on S and T alike.

## Runs (toys/ts_toy.py: idea12_toy's store/replay plus vig_toy_sleep's sleep; 500-line store)
0. references: no replay; sleep=1000 (+pairs); match=4 + targeted 0.25 (must reproduce 0.93).
1. C: tag=1 with nov=1, nov=2, nov=0 control; tagB 0/1; with and without match=4; with and without sleep=1000.
2. D: sleep=1000 with ds_frac 0.3 / 0.6, ds_sub 0.5; ds_all 0/1; with and without match=4 and pairs.
3. C+D: the best C with the best D, +/- match=4.
At most a few dozen toy runs (each ~3-6 min, 4 in parallel). The novelty trace per phase is logged to check the falsifier.

## Full size (only if a toy passes)
New script t23_life_ts.py (copy of t23_life_idea12.py importing net12_ts), cue=1 seed 1, K8 lagro=1 relay=1 + the passing options.
Compare with LIFE_RULES_REPORT.md: succ 0.98 -> 0.07, text 0.35 -> 0.24, bits 828,527. Full-size success: succ at end >= 0.5,
copy/rev >= 0.95, bits within 2% of 828,527. ~30-60 min per run.
NOT_RUN by design: other seeds, cue=0, any search over the time constants beyond the few listed.
