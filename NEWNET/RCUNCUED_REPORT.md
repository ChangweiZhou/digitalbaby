# Rule cells that also read match cells, without cues (23 November 2026)

Branch `claude/newnet-rcuncued-sei232` from `claude/magical-albattani-sei232` @ d0f0a9da. Development level, one machine, **seed 1
only**. No gate, seal or outside review is claimed. The plan and predictions (`RCUNCUED_PLAN.md`) were committed before any run.
New files only: `net13_m.py` (subclass of net13, option `rc_match=K`), `net131.py` (alias so the life scripts load it as net=131),
`check_net13_m.py`, `rcuncued_runs/` (run.sh, logs, res_t23.jsonl, res_t27.jsonl, table.py -> table.md). No existing .py changed.
Learning is local throughout. 8 lives, about 26-28 min each on one core, 4 at a time (~3.6 CPU-hours).

## The change
The rule cells' input set becomes lag cells + match cells (m[k, j] = 1 when the byte k back equals the byte k+j back, k = 0..3,
j = 1..7-k: 22 cells that can fire). x stays normalised (each active cell 1/sqrt(n)). The cosine competition, recruitment, persistence,
k-winner and prototype learning are unchanged. The gate, the direct readout and net10's compartments read what they read in net13.
Checks:
- `check_net13_m.py 300` (3,600 bytes, 400 with learning off): rc_match=0 against net13 gives max|dO| **0** and **0** differing answers.
  rc_match=4 against net13 gives 462 differing answers, so the option is active.
- net13 here reproduces the earlier numbers exactly: t23 cue=1 777,024 bits (NET13_REPORT) and t27 cue=0 / cue=1 809,317 / 812,727
  bits (RCNEWLIFE, net12_rc).

## Results (seed 1)
"end" = the last probe (line 29,000, learning off, 60 fresh items a rule, teacher-forced answer bytes; with cue=0 all items use one
delimiter).

| life | net | cue | bits total | answer bits | end probe | mapping: after its last teaching -> end |
|---|---|---|---|---|---|---|
| t23 | net13 | 0 | 785,622 | 80,657 | copy 0.65 / rev 1.00 / **succ 0.02** / text 0.29 | 0.99 -> 0.02 |
| t23 | net13 + rc_match=4 | 0 | 784,339 | 80,061 | copy 0.34 / rev 1.00 / **succ 0.02** / text 0.29 | 0.98 -> 0.02 |
| t23 | net13 | 1 | 777,024 | 68,398 | 1.00 / 1.00 / **0.82** / 0.24 | 1.00 -> 0.82 |
| t23 | net13 + rc_match=4 | 1 | 781,870 | 70,266 | 1.00 / 1.00 / **0.76** / 0.26 | 0.99 -> 0.76 |
| t27 | net13 | 0 | 809,317 | 96,393 | copy 0.37 / swap 1.00 / rot 0.37 / skip 0.06 / **map 0.01** / text 0.30 | 0.94 -> 0.01 |
| t27 | net13 + rc_match=4 | 0 | 808,929 | 94,123 | 0.38 / 1.00 / 0.37 / 0.46 / **0.02** / 0.28 | 0.93 -> 0.02 |
| t27 | net13 | 1 | 812,727 | 86,930 | 1.00 / 1.00 / 1.00 / 1.00 / **0.88** / 0.29 | 0.99 -> 0.88 |
| t27 | net13 + rc_match=4 | 1 | 809,751 | 85,462 | 1.00 / 1.00 / 0.99 / **0.52** / **0.90** / 0.30 | 0.99 -> 0.90 |

- **Without cues, the mapping is gone within 1,000 lines of the next rule, in both nets.** This holds for t23 succ (0.98-0.99 at
  23k, then 0.00 at 24k) and for t27 map (0.93-0.94 at 23k, then 0.00-0.01 at 24k).
- The full probe series, first-met right shares by phase and recruit counts are in `rcuncued_runs/table.md`.
- First-met learning is unchanged within ±0.04 in every phase. For example, t23 cue=0 P3 succ b is 0.68 against 0.69. t27 cue=0 P1
  map b is 0.45 against 0.49, and P6 map b is 0.61 against 0.60.
- Recruits on t27: cue=0 37,930 against 15,893; cue=1 54,350 against 74,117.
- Side effects of rc_match at the end, not predicted:
  - t27 cue=0: skip is better, 0.46 against 0.06.
  - t23 cue=0: copy is worse, 0.34 against 0.65.
  - t27 cue=1: skip is worse, 0.52 against 1.00.

## Verdict against the predictions

| prediction | needed | got | met? |
|---|---|---|---|
| 1. t23 cue=0 succ at the end | >= 0.5 (falsifier < 0.3) | 0.02 (net13 0.02) | **no: falsified** |
| 2. t27 cue=0 map at the end | >= 0.5 (falsifier < 0.3) | 0.02 (net13 0.01) | **no: falsified** |
| 3a. t23 cue=1 | succ >= 0.80, bits <= 784,800 | succ 0.76, 781,870 bits | **no** (succ) |
| 3b. t27 cue=1 | map >= 0.85, bits <= 820,854 | map 0.90, 809,751 bits | yes; but skip falls to 0.52 |
| 4. cue=0 bits lower than net13 | lower on both | 784,339 vs 785,622 (-0.16%); 808,929 vs 809,317 (-0.05%) | yes, by a negligible margin |

**Plain verdict: the hypothesis is refuted for this implementation.** When the rule cells read match cells as well as lag cells,
they do not keep the learned mapping without cues. On both lives it is lost as completely as in net13 (0.02 at the end). With cues,
the change is mixed: t23 succ is slightly worse (0.76 against 0.82), t27 map is slightly better (0.90 against 0.88), and t27 skip is
much worse (0.52 against 1.00). The bit differences are all within 0.6%.

The plan required seed 1 to pass before seeds 2-3, so **seeds 2-3 were not run**.

## Possible reasons (not tested here)
These are readings of the design, not measurements.
- **Match cells are a small part of x.** Only a few match cells fire against 8 active lag cells, so the cosine is still decided mainly by
  content. A mapping's answers (succ, map) also make no matches, the same as most input bytes.
- **The lag cells' gate decides before the rule is visible.** It sets the first answer byte before any answer byte has been read, and
  that gate is shared across rules. In the earlier diagnosis this shared gate is what the next rule's training reopens. Better
  inference of the rule in the rule cells does not protect it.
- **Rule-cell activity does not last across lines.** It decays with tau=8 under k=3 competition, so a rule inferred on one line's
  answers does not carry to the next line's answers.

## Limits / NOT_RUN
- Seed 1 only. Not run: seeds 2-3, other K values, other weightings of the match cells, toys, and battery tests.
- I did not log which rule cells won on which answer bytes, so the reasons above are not verified.
