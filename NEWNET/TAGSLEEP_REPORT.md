# Tag-and-capture by novelty (idea C) and sleep downscaling (idea D) (23 November 2026)

Helper branch `claude/newnet-tagsleep-sei232`, from `claude/magical-albattani-sei232` @ 9dcc213e. Development level, one machine,
**seed 1 only**; no gate, seal or outside review is claimed. Plan and criteria written before the runs: `TAGSLEEP_PLAN.md`.
Code (new files only; no existing .py edited): `net12_ts.py` (subclass of `net12_rep.Tissue`), `toys/ts_toy.py`, `t23_life_ts.py`
(copy of `t23_life_idea12.py` importing net12_ts, with `sleep=N`). Logs: `tagsleep_runs/` (`*.out` toys, `life_*.log`,
`life_res.jsonl`). Learning stays local (net12's delta rules); store = 500 episodes.

## Mechanisms as run
- **C (tag=1)**: relay-gate strength (tagB=1: also the direct readout) = S + T. net12's change goes into T; each byte read with
  learning on, T decays (tag_tau = 300 bytes; one run 3,000) and cap * nov of it (cap = 0.02) is captured into S. Novelty: nov=1
  error (fast 50-byte mean of 1 - O[byte] over its 5,000-byte mean, / 0.2, clipped 0..1); nov=2 input (1 - cos of fast vs slow byte
  traces over its own mean); nov=0 constant 0.1 (control). rep_nov=1: novelty 1 while a stored line is replayed.
- **D**: at the start of each sleep (after each phase, before 1,000 stored lines are replayed offline): ds_frac (G *= 1 - f) or
  ds_sub (|G| -= s * mean of nonzero |G|, floored at 0); ds_all=1: also B and net10's readouts. Checked that it acts: after 1,500
  copy lines ds_sub=0.5 removes 2,034 of 2,624 nonzero gate strengths (1.0: 2,180), ds_frac=0.3 scales the max 2.02 -> 1.41; the
  copy probe stays 1.00 in all three.

## Toy (4 regions, copy 2000 -> rev 2000 -> succ 3000 -> copy 1000, cued; probes with learning off after each phase and its sleep)
Criterion (fixed before): succ after the last copy >= 0.85, copy and rev there >= 0.95, first-200 copy (phase 1) >= 0.6.
nov = mean novelty over the online lines of the succ / last copy phases. m4 = match=4; mt = match=4 + targeted replay 1 per 4;
sl = sleep=1000 (random draw unless tgt=1).

| run | options | first200 copy | after succ: copy / rev / succ | last copy first200 | after last copy: copy / rev / **succ** | nov succ / last copy | met |
|---|---|---|---|---|---|---|---|
| ref_none | - | 0.73 | 0.53 / 0.97 / 1.00 | 0.79 | 1.00 / 1.00 / **0.58** | 0.173 / 0.013 | no |
| ref_sleep | sl | 0.73 | 0.92 / 1.00 / 0.94 | 0.97 | 1.00 / 1.00 / **0.71** | 0.184 / 0.006 | no |
| ref_sleep_p | sl + pairs | 0.69 | 0.98 / 1.00 / 0.85 | 0.99 | 1.00 / 1.00 / **0.72** | 0.205 / 0.000 | no |
| ref_sleep_m4 | sl + m4 | 0.70 | 0.96 / 1.00 / 0.97 | 0.99 | 1.00 / 1.00 / **0.82** | 0.183 / 0.004 | no |
| ref_sleep_m4_tgt | sl tgt=1 + m4 | 0.70 | 0.81 / 0.99 / 1.00 | 0.90 | 1.00 / 1.00 / **0.95** | 0.186 / 0.029 | yes |
| ref_mt | mt | 0.67 | 0.81 / 0.98 / 1.00 | 0.87 | 1.00 / 1.00 / **0.93** | 0.147 / 0.014 | yes |
| ref_mt_sleep | mt + sl | 0.67 | 0.77 / 0.98 / 0.98 | 0.90 | 1.00 / 1.00 / **0.93** | 0.156 / 0.053 | yes |
| C_n1 | tag nov=1 | 0.70 | 0.07 / 0.35 / 1.00 | 0.32 | 0.97 / 0.66 / **0.69** | 0.101 / 0.055 | no |
| C_n2 | tag nov=2 | 0.70 | 0.17 / 0.37 / 1.00 | 0.49 | 1.00 / 0.77 / **0.38** | 0.053 / 0.281 | no |
| C_n0 | tag nov=0 (const 0.1) | 0.58 | 0.21 / 0.51 / 1.00 | 0.47 | 1.00 / 0.94 / **0.62** | 0.100 / 0.100 | no |
| C_n1_B | tag tagB nov=1 | 0.69 | 0.88 / 0.72 / 0.66 | 0.95 | 0.98 / 0.59 / **0.14** | 0.091 / 0.001 | no |
| C_t3k | tag nov=1 tag_tau 3000 | 0.73 | 0.23 / 0.45 / 1.00 | 0.59 | 1.00 / 0.88 / **0.62** | 0.129 / 0.031 | no |
| C_n1_m4 | tag + m4 | 0.64 | 0.14 / 0.35 / 1.00 | 0.47 | 0.99 / 0.64 / **0.79** | 0.102 / 0.044 | no |
| C_n1_mt | tag + mt | 0.63 | 0.34 / 0.40 / 1.00 | 0.64 | 0.99 / 0.77 / **0.82** | 0.094 / 0.041 | no |
| C_n1_sl | tag + sl | 0.70 | 0.43 / 0.56 / 0.99 | 0.72 | 0.97 / 0.79 / **0.62** | 0.122 / 0.035 | no |
| CR_sl | tag rep_nov + sl | 0.70 | 0.88 / 0.84 / 0.90 | 0.97 | 0.99 / 0.99 / **0.61** | 0.109 / 0.003 | no |
| D_f3 | sl ds_frac .3 | 0.73 | 0.68 / 0.91 / 0.98 | 0.89 | 0.99 / 0.98 / **0.66** | 0.161 / 0.018 | no |
| D_f6 | sl ds_frac .6 | 0.73 | 0.45 / 0.62 / 0.99 | 0.76 | 0.98 / 0.73 / **0.66** | 0.140 / 0.027 | no |
| D_f3_all | sl ds_frac .3 ds_all | 0.73 | 0.88 / 0.99 / 0.90 | 0.97 | 1.00 / 0.99 / **0.42** | 0.150 / 0.009 | no |
| D_f3_p | sl ds_frac .3 + pairs | 0.69 | 0.96 / 1.00 / 0.86 | 0.98 | 1.00 / 0.99 / **0.68** | 0.181 / 0.008 | no |
| D_s5 | sl ds_sub .5 | 0.73 | 0.91 / 1.00 / 0.94 | 0.98 | 1.00 / 1.00 / **0.73** | 0.182 / 0.007 | no |
| D_s10 | sl ds_sub 1 | 0.73 | 0.92 / 1.00 / 0.94 | 0.98 | 1.00 / 1.00 / **0.73** | 0.181 / 0.007 | no |
| D_f3_m4 | sl ds_frac .3 + m4 | 0.70 | 0.79 / 0.96 / 0.98 | 0.93 | 1.00 / 0.99 / **0.74** | 0.161 / 0.014 | no |
| D_s5_m4 | sl ds_sub .5 + m4 | 0.70 | 0.96 / 1.00 / 0.97 | 0.99 | 1.00 / 1.00 / **0.83** | 0.181 / 0.004 | no |
| D_s10_m4 | sl ds_sub 1 + m4 | 0.70 | 0.96 / 1.00 / 0.97 | 0.99 | 1.00 / 1.00 / **0.83** | 0.180 / 0.004 | no |
| D_s5_m4_tgt | sl tgt=1 ds_sub .5 + m4 | 0.70 | 0.79 / 0.97 / 0.98 | 0.91 | 1.00 / 0.99 / **0.94** | 0.184 / 0.029 | yes |
| D_s5_mt | mt + sl ds_sub .5 | 0.67 | 0.80 / 0.97 / 0.99 | 0.89 | 0.99 / 0.99 / **0.94** | 0.155 / 0.046 | yes |
| CD_f3 | tag + sl ds_frac .3 | 0.70 | 0.23 / 0.43 / 0.99 | 0.51 | 0.85 / 0.62 / **0.63** | 0.111 / 0.048 | no |
| CD_f3_m4 | tag + sl ds_frac .3 + m4 | 0.64 | 0.28 / 0.44 / 0.99 | 0.54 | 0.88 / 0.63 / **0.72** | 0.112 / 0.045 | no |
| CR_s5 | tag rep_nov + sl ds_sub .5 | 0.70 | 0.87 / 0.84 / 0.90 | 0.96 | 0.99 / 0.98 / **0.63** | 0.109 / 0.004 | no |
| CR_s5_m4 | the same + m4 | 0.64 | 0.92 / 0.89 / 0.94 | 0.96 | 1.00 / 1.00 / **0.71** | 0.107 / 0.001 | no |
| C_t3k_sl | tag tau 3000 rep_nov + sl ds_sub .5 | 0.73 | 0.91 / 0.97 / 0.94 | 0.98 | 1.00 / 1.00 / **0.67** | 0.135 / 0.001 | no |

References reproduce the earlier numbers (no replay 0.58; match=4 + targeted 0.93; sleep + pairs 0.72, earlier 0.72).

What the toy shows:
1. **C fails, and not for the expected reason.** The falsifier for C was a novel copy return; it did not happen: error novelty during
   the last copy is 0.001-0.055 (in its first 200 lines at most 0.26), so copy's changes there do fade. But tags also fade whatever was
   learned after a rule's error had dropped, and capture during the novel succ phase fixes the gate changes that close copy: copy
   after succ falls from 0.53 to 0.07-0.34 (without sleep), rev is learned only to 0.84-0.92, and the last copy has to relearn
   (first200 0.32-0.64). Succ after the last copy 0.38-0.82. Without replay, C raises it a little (0.58 -> 0.62-0.69; with m4 0.79, no m4-only
   reference was run) only by losing copy and rev (C_n1: copy 0.07 after succ, rev 0.66 at the end); with replay or sleep every C
   variant is below its reference without C. Input novelty (nov=2) is high during the copy return (0.28)
   and is worse (0.38). Tagging the direct readout too (tagB) loses the succ mapping itself (0.14). Replay-gated capture (rep_nov)
   with sleep restores copy/rev but succ stays 0.61-0.71.
2. **D adds at most +0.02.** Downscaling changes succ after the last copy by -0.29..+0.02 against the same sleep without it
   (sleep 0.71 -> ds_sub 0.73, ds_frac 0.66, ds_all 0.42; with m4 0.82 -> 0.83 / 0.74; with m4 + targeted sleep 0.95 -> 0.94;
   with mt + sleep 0.93 -> 0.94). Subtractive downscaling removes ~80% of the nonzero gate strengths, yet replay rebuilds the same
   outcome; multiplicative downscaling hurts.
3. **C+D**: 0.63-0.72, below D alone.
4. Two runs with D meet the criterion (D_s5_m4_tgt 0.94, D_s5_mt 0.94), but their references without D meet it as well (0.95, 0.93):
   the pass comes from match=4 + targeted replay (online, or during sleep: **targeted sleep + match=4 without D, 0.95**, a new
   variant of the known passing mechanism), not from C or D.

## Full size (t23_life_ts.py, K8 net12 lagro=1 relay=1 match=4, cue=1, seed 1; ~26-32 min each, 4 in parallel)
Run because two toy configurations passed; each with its reference. Probes: after text = line 17,000, after succ = 23,000, after
copy = 26,000 (before P4's sleep), end = 29,000 (before P5's sleep). Full-size criterion (plan): succ at end >= 0.5, copy/rev >= 0.95,
bits within 2% of 828,527.

| t23, cue=1, seed 1 | bits total (text part) | text after text -> end | succ after succ -> after copy -> end | copy / rev at end | P0 copy b / c | P3 succ b | met |
|---|---|---|---|---|---|---|---|
| net12 relay, no replay (LIFE_RULES_REPORT.md) | 828,527 (146,104) | 0.35 -> 0.24 | 0.98 -> 0.17 -> 0.07 | 1.00 / 1.00 | 0.98 / 0.73 | 0.51 | no |
| m4 + sleep 1000 targeted | 837,688 (149,056) | 0.36 -> 0.26 | 0.98 -> 0.41 -> 0.33 | 1.00 / 1.00 | 0.97 / 0.73 | 0.52 | no |
| m4 + sleep 1000 targeted + ds_sub .5 (D) | 836,999 (148,905) | 0.35 -> 0.28 | 0.98 -> 0.44 -> 0.33 | 1.00 / 1.00 | 0.97 / 0.73 | 0.53 | no |
| m4 + targeted 1 per 4 (online) | 862,949 (165,224) | 0.35 -> 0.33 | 0.99 -> 0.52 -> 0.38 | 1.00 / 1.00 | 0.98 / 0.73 | 0.53 | no |
| m4 + targeted 1 per 4 + sleep + ds_sub .5 (D) | 872,620 (167,774) | 0.35 -> 0.30 | 0.98 -> 0.59 -> 0.39 | 1.00 / 1.00 | 0.98 / 0.73 | 0.54 | no |

Novelty (error) per phase in the sleep runs: copy 0.065, rev 0.086, text 0.196, succ 0.075, copy 0.001, rev 0.002 (D did not run C,
the signal is only logged). First window when copy / rev return: 0.97-0.99 / 1.00 in all.
No run meets the full-size criterion: succ at end 0.33-0.39 (< 0.5). Sleep with targeted draw + match=4 keeps the bits within 2%
(+1.1%; online replay costs +4.2-5.3%, mostly in the text phase) and moves succ at end from 0.07 to 0.33. Downscaling changes
succ at end by -0.005 / +0.006 and bits by -0.1% / +1.1% against its reference.

## Verdict (seed 1)
Neither idea solves the loss of the learned mapping. **C (tag-and-capture gated by novelty) is refuted on the toy**: copy's
return is indeed low in novelty, but the same rule erases what the gate learns after each rule's error falls and fixes succ's
gate changes that close copy, so copy and rev are lost instead (best C run 0.82 with match + targeted replay, below that reference's
0.93). **D (sleep downscaling) is neutral**: it acts (removes ~80% of the nonzero gate strengths) but replay restores the same
state; +/-0.02 on the toy, +/-0.01 at full size. The only toy passes come from match=4 + targeted replay, online (0.93) or, new
here, in sleep (0.95, no D). At full size these keep succ at the end at 0.33-0.39 (from 0.07), short of 0.5; targeted sleep is the
cheap form (+1.1% bits vs +4.2% for online replay).
NOT_RUN: other seeds, cue=0, other cap / tag_tau / novelty scales beyond those listed, C at full size (no C passed the toy),
a probe after each sleep at full size.
