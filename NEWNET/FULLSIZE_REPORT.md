# Full-size lives for the three toy passes (t23, net12 K8 lagro=1 relay=1 cue=1, seed 1; 23 November 2026)

Development level, one machine, **seed 1 only**; no gate, seal or outside review is claimed. Source: branch
`claude/magical-albattani-sei232` at d169be40 (scripts unchanged), run on `claude/newnet-fullsize-sei232`, OMP_NUM_THREADS=1, the
three lives run at once. Logs: `fullsize_runs/*_s1.log`; results: `fullsize_runs/res.jsonl`. Earlier attempts killed by restarts of
another machine are kept in `combo_runs/*_killed_by_restart*.log` (not used here). Life as in LIFE_RULES_REPORT.md:
copy 6,000 -> rev 6,000 -> 40,000 text bytes -> succ 6,000 -> copy 3,000 -> rev 3,000 lines (29,000 probe lines in all).

Commands (in NEWNET/):
- A: `t23_life_idea12.py net=12 struct=K8 lagro=1 relay=1 match=4 rho=0.25 tgt=1 cue=1` (match cells + targeted replay), 1,661 s
- B: `t23_life_rules.py net=12 struct=K8 lagro=1 relay=1 recall=500 cue=1` (episodic recall compartment), 1,353 s
- C: `t23_life_rules.py net=12 struct=K8 lagro=1 relay=1 recall=500 match=4 cue=1` (recall + match cells), 1,363 s

## Results

| seed 1, cue=1 | bits total | text part | P0 copy b / c | P1 rev b | P3 succ b | P4 copy b | P5 rev b | text 17000 -> 29000 | succ 23000 -> 26000 -> 29000 | copy / rev at 29000 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (LIFE_RULES_REPORT.md) | 828,527 | 146,104 | 0.98 / 0.73 | 0.80 | 0.51 | 0.99 | 1.00 | 0.35 -> 0.24 | 0.98 -> 0.17 -> 0.07 | 1.00 / 1.00 |
| A match=4 + targeted replay | 862,949 | 165,224 | 0.98 / 0.73 | 0.81 | 0.53 | 1.00 | 1.00 | 0.35 -> **0.33** | 0.99 -> 0.52 -> **0.38** | 1.00 / 1.00 |
| B recall=500 | 833,200 | 145,112 | 0.97 / 0.42 | 0.81 | 0.62 | 0.99 | 1.00 | 0.35 -> 0.25 | 0.93 -> 0.75 -> **0.17** | 1.00 / 1.00 |
| C recall=500 + match=4 | 831,453 | 146,526 | 0.97 / 0.40 | 0.81 | 0.64 | 0.99 | 1.00 | 0.36 -> 0.27 | 0.93 -> 0.78 -> **0.36** | 1.00 / 1.00 |

(first-met = share of first-met answer bytes right; layer b new combinations, c never-answered symbols, 40 bytes.)

Notes from the logs:
- A's replay counts per phase (log): P3 succ 714 / copy 914 / rev 741 / text 381; P4 succ 423; P5 succ 482. The text bits rise by
  19,120 (+13%); the text probe at the end is the best of the four (0.33).
- B and C keep succ through the returning copy (0.75 / 0.78 at 26000, baseline 0.17), but it falls during the returning rev
  (C: 0.67 at 27000, 0.44 at 28000, 0.36 at 29000). Both learn succ faster (P3 b 0.62 / 0.64 against 0.51), but halve the first-met
  never-answered symbols of P0 copy (c 0.42 / 0.40 against 0.73). During succ the copy probe dips to 0.92 (line 23000), then recovers.
- All three keep copy and rev at 1.00 at the end and cost at most +4.2% bits (A); B and C cost +0.6% / +0.4%.

## Verdict
Plain: **none of the three toy passes holds at full size.** Every run keeps more of succ at the end than the baseline (0.38, 0.17,
0.36 against 0.07) and copy/rev stay at 1.00, but none reaches the toy criterion (succ after the last copy >= 0.85; here even the
helper's lower bar of succ at the end >= 0.5 is missed), so seeds 2 and 3 were **NOT_RUN** as instructed. The recall compartment
protects succ against the returning copy (0.75-0.78 after copy) but not against the following rev; match + targeted replay keeps the
text best (0.33) at +13% text bits. E2 for a learned mapping and for text remains open. Seed 1 only.
