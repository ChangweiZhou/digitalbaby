# Idea 12: targeted replay and slow consolidated gate strengths (23 November 2026)

Helper branch `claude/newnet-idea12-sei232` (from `claude/magical-albattani-sei232` @ 6dc127e8). Development level, one machine,
**seed 1 only**; no gate, seal or outside review is claimed. Plan and criteria written before the runs: `IDEA12_PLAN.md`.
Code (new files only): `net12_rep.py` (net12 + fast/slow lag strengths), `toys/idea12_toy.py` (4-region toy with store, random or
targeted replay), `t23_life_idea12.py` (t23 life + the same store/replay). Logs: `idea12_runs/`. Learning stays local (net12's delta
rules); store = 500 episodes (8-byte rule lines; 40-byte text episodes in the life).

## Mechanisms as run
- Targeted replay (tgt=1): each stored line keeps its last error (mean over its answer bytes of 1 - p(answer)), set when read,
  updated when replayed; drawn with probability proportional to error^2 (+1e-4).
- Slow gate (slow=1, the owner's form): relay-gate strength = fast + slow; on each replayed byte slow += sk * fast (fast *= 1 - sk);
  the fast part decays toward 0 with time constant ftau bytes. slow=2: replay's own changes go into the slow part. slowB=1: the same
  for the direct readout. Without the decay the split cannot change anything: checked, slow=1 ftau=0 + targeted gives exactly the
  targeted run's numbers (all phases).

## Toy (4 regions, life copy 2000 -> rev 2000 -> succ 3000 -> copy 1000, cued; probes with learning off after each phase)
first200 = right share in the phase's first 200 lines; probes copy/rev/succ. The toy stores a line after it is read (the older
`vig_toy_replay.py` stored it before, so it could be replayed just before being read); its random-0.25 result there was 0.86 / 0.68
(reproduced here), in the new script 0.83 / 0.71.

| run | copy first200 | rev first200 | succ first200 | after succ: copy / rev / succ | final copy first200 | after final copy: copy / rev / **succ** |
|---|---|---|---|---|---|---|
| random 0.25 | 0.71 | 0.35 | 0.01 | 0.83 / 0.98 / 0.97 | 0.94 | 1.00 / 1.00 / **0.71** |
| random 0.5 | 0.74 | 0.35 | 0.01 | 0.96 / 0.99 / 0.95 | 0.96 | 1.00 / 1.00 / **0.69** |
| targeted 0.25 | 0.72 | 0.35 | 0.01 | 0.66 / 0.98 / 0.99 | 0.85 | 1.00 / 1.00 / **0.75** |
| slow=1 sk .05 ftau 500 + random 0.25 | 0.67 | 0.34 | 0.00 | 0.79 / 0.98 / 0.97 | 0.93 | 1.00 / 1.00 / **0.71** |
| slow=1 sk .05 ftau 500 + targeted 0.25 | 0.69 | 0.34 | 0.01 | 0.61 / 0.96 / 0.99 | 0.84 | 1.00 / 1.00 / **0.79** |
| slow=1 sk .01 ftau 2000 + targeted | 0.69 | 0.34 | 0.01 | 0.61 / 0.94 / 0.99 | 0.83 | 1.00 / 1.00 / 0.77 |
| slow=1 sk .2 ftau 200 + targeted | 0.69 | 0.34 | 0.00 | 0.60 / 0.95 / 0.99 | 0.85 | 1.00 / 1.00 / 0.78 |
| slow=1 ftau 0 (no decay) + targeted | 0.72 | 0.35 | 0.01 | 0.66 / 0.98 / 0.99 | 0.85 | 1.00 / 1.00 / 0.75 (= targeted) |
| slow=2 ftau 500 + random | 0.56 | 0.34 | 0.00 | 0.64 / 0.53 / 0.92 | 0.87 | 1.00 / 0.77 / 0.47 |
| slow=2 ftau 500 + targeted | 0.58 | 0.34 | 0.00 | 0.54 / 0.52 / 0.97 | 0.76 | 0.99 / 0.81 / 0.53 |
| slow=2 ftau 5000 + targeted | 0.70 | 0.35 | 0.00 | 0.47 / 0.41 / 0.99 | 0.79 | 1.00 / 0.72 / 0.36 |
| slow=2 slowB=1 ftau 500 + random | 0.48 | 0.38 | 0.00 | 0.89 / 0.81 / 0.10 | 0.97 | 1.00 / 0.86 / 0.11 |
| (no replay, earlier report) | | | | 0.53 / - / - | | - / - / 0.58 |

Against the criteria (succ after final copy >= 0.85 with copy/rev >= 0.95 at rho <= 0.25): **no run succeeds**. Best: slow=1 +
targeted 0.79 (+0.08 on random 0.25: below the +0.10 "partial" bar), and it pays for it where the store is replayed for succ: copy
after succ 0.61 (random 0.83) and the final copy's first200 0.84 (random 0.94, a drop of 0.10). Targeted replay alone: +0.04 on succ,
-0.17 on copy after succ (it replays the line kind now worst, so the rule not in training is starved until its error shows). Doubling
random replay (0.5) helps copy after succ (0.96) but not succ after copy (0.69). Slow=2 and slowB=1 learn less and forget more.

## Full size (t23 life, K8 net12 lagro=1 relay=1, cue=1, seed 1, rho 0.25; ~28 min each, 3 in parallel)

| t23, cue=1, seed 1 | bits total (text part) | text probe after text -> end | succ after succ -> after copy -> end | copy / rev at end | P3 succ first-met b |
|---|---|---|---|---|---|
| no replay (LIFE_RULES_REPORT.md) | 828,527 (146,104) | 0.35 -> 0.24 | 0.98 -> 0.17 -> 0.07 | 1.00 / 1.00 | 0.51 |
| random replay 0.25 | 860,360 (162,722) | 0.35 -> 0.29 | 0.95 -> 0.21 -> 0.12 | 1.00 / 1.00 | 0.45 |
| targeted replay 0.25 | 865,721 (164,705) | 0.34 -> 0.32 | 0.97 -> 0.37 -> 0.23 | 1.00 / 1.00 | 0.50 |
| slow=1 (sk .05, ftau 500) + targeted 0.25 | 868,951 (163,259) | 0.36 -> 0.34 | 0.96 -> 0.31 -> 0.22 | 1.00 / 1.00 | 0.48 |

(after copy = probe at line 26,000, the end of P4; end = line 29,000, after the final rev.) First-met P0 copy b/c 0.98 / 0.72-0.73
in all; returns of copy and rev right at once (first window 0.97-0.99, 1.00).
Full-size criterion (succ at end >= 0.5, copy/rev >= 0.95, bits within 2% of 828,527): **not met**. Replay keeps the text better
(probe at end 0.29-0.34 against 0.24) and the learned mapping a little better (0.12-0.23 against 0.07), but every replay run costs
3.8-4.9% more bits (most of it during the text phase: 4.07-4.12 bits per text byte against 3.65; replayed rule lines interleaved
with the text disturb its prediction). The slow gate adds nothing at full size over targeted replay alone (0.22 vs 0.23 succ at end).

## Verdict (seed 1)
Neither targeted replay nor slow consolidated gate strengths solves the loss of the learned mapping. In the toy the best combination
(slow transfer + targeted) moves succ after the final copy from 0.71 (random replay) to 0.79, short of 0.85, and costs copy (0.61
after succ). At full size replay of a 500-episode store at 1 per 4 lines keeps text and succ somewhat better (succ at end 0.07 ->
0.23 with targeted replay, text 0.24 -> 0.32) but costs ~4.5% bits, and succ still falls from 0.97 to 0.37 within the 3,000 copy
lines. The slow part, as specified, only acts through the fast part's decay; with decay it either changes little (slow=1) or
forgets what was not replayed (slow=2). The interference is not slowed enough by rehearsing a small store at this rate.
NOT_RUN: other seeds, cue=0, rho 0.5 at full size, other error exponents or a top-fraction rule for targeted replay.
