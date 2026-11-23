# Rule cells on seeds 2-3, and rule cells + targeted replay (23 November 2026)

Helper branch `claude/newnet-rcseeds-sei232` (from `claude/magical-albattani-sei232` @ 6cf09658). Development level, one machine,
cue=1 only. **No gate, seal or outside review is claimed.** Every number below is from `rcseeds_runs/res.jsonl` (this branch) and the
earlier results files named; the table is printed by `rcseeds_runs/table.py`. OMP_NUM_THREADS=1, 4 then 2 processes.

## What was run
- **rc**: the t23 life (cue=1), `t23_life_rules_rc.py net=12 struct=K8 lagro=1 relay=1 cue=1 rc=64`, rc_gate=1 by default.
  This is the same driver and options as RULECELLS_REPORT.md run A (`rc_runs/run_full.sh`). New here: seeds 2 and 3. Seed 1 is from
  `rc_runs/full_res.jsonl`.
- **base**: net12 lagro=1 relay=1. Seeds 2-3 were already in `seeds23_runs/res_t23.jsonl` with the same command, so they were not
  re-run. Seed 1 is from `life_rules_runs/res.jsonl`.
- **rc+rep**: rule cells plus targeted replay.
  - Store: 500 episodes, reservoir.
  - Replay: one episode every 4 lines (8 text bytes count as a line), drawn in proportion to its last error^2, with learning on.
  - This is IDEA12's `life_tgt25`, with rc=64 added.
  - Code (new files only; no existing .py was edited):
    - `net12_rcrep.py`: `class Tissue(net12_rep.Tissue, net12_rc.Tissue)`. The MRO is rep -> rc -> net12. With slow=0, rep only
      adds the `replaying` flag.
    - `t23_life_idea12_rc.py`: `t23_life_idea12.py` with that import changed. Nothing else differs.
  - Seed 1 ran first. Its end values were succ 0.706 >= 0.7 and text 0.319 >= 0.30, so seeds 2 and 3 were run.
- **Check**: `t23_life_idea12_rc.py ... rc=64` with rho=0 (no replay), seed 1. It reproduces run A exactly: 777,024 bits (text
  143,714), with the same first-met numbers and probes (`rcseeds_runs/rcrep0_s1_check.log`).
- Cost: about 25 min per rc life and about 32 min per rc+rep life, on one core each.

## Per run
- Bits total (text part) are over the whole life.
- b P0..P5 is the right share on first-met answer bytes of layer b in each phase: P0 copy, P1 rev, P3 succ, P4 copy, P5 rev.
- c is P0 copy layer c, symbols never given as an answer.
- Probes are taken with learning off. Text is at line 17,000 (end of the text phase) -> 29,000 (end). Succ is at 23,000 (end of
  succ) -> 26,000 (after the copy phase) -> 29,000 (after the final rev).

| run | seed | bits total (text) | b P0 / P1 / P3 / P4 / P5 | c P0 | text 17k -> 29k | succ 23k -> 26k -> 29k | copy / rev 29k |
|---|---|---|---|---|---|---|---|
| base | 1 | 828,527 (146,104) | 0.977 / 0.795 / 0.505 / 0.991 / 1.000 | 0.725 | 0.351 -> 0.242 | 0.983 -> 0.167 -> 0.067 | 1.00 / 1.00 |
| base | 2 | 829,414 (147,252) | 0.976 / 0.797 / 0.506 / 0.988 / 1.000 | 0.725 | 0.370 -> 0.260 | 0.950 -> 0.150 -> 0.028 | 1.00 / 1.00 |
| base | 3 | 831,508 (149,321) | 0.979 / 0.794 / 0.500 / 0.988 / 1.000 | 0.725 | 0.347 -> 0.228 | 0.983 -> 0.139 -> 0.039 | 1.00 / 1.00 |
| rc | 1 | 777,024 (143,714) | 0.995 / 0.948 / 0.715 / 1.000 / 1.000 | 0.825 | 0.348 -> 0.240 | 1.000 -> 0.983 -> 0.822 | 1.00 / 1.00 |
| rc | 2 | 776,645 (144,619) | 0.996 / 0.956 / 0.711 / 1.000 / 1.000 | 0.825 | 0.340 -> 0.250 | 0.978 -> 0.950 -> 0.789 | 1.00 / 1.00 |
| rc | 3 | 777,800 (146,519) | 0.997 / 0.953 / 0.733 / 1.000 / 1.000 | 0.800 | 0.329 -> 0.241 | 0.994 -> 0.917 -> 0.822 | 1.00 / 1.00 |
| rc+rep | 1 | 825,552 (164,829) | 0.995 / 0.955 / 0.690 / 0.985 / 0.984 | 0.825 | 0.340 -> 0.319 | 1.000 -> 0.889 -> 0.706 | 1.00 / 1.00 |
| rc+rep | 2 | 825,591 (165,135) | 0.995 / 0.958 / 0.699 / 0.991 / 0.989 | 0.825 | 0.335 -> 0.298 | 0.994 -> 0.833 -> 0.578 | 1.00 / 1.00 |
| rc+rep | 3 | 824,072 (165,968) | 0.996 / 0.954 / 0.694 / 0.985 / 0.989 | 0.825 | 0.304 -> 0.280 | 0.994 -> 0.933 -> 0.539 | 1.00 / 1.00 |

For comparison, seed 1 with targeted replay but no rule cells (IDEA12_REPORT.md, `idea12_runs/life_res.jsonl`):
- bits 865,721 (164,705)
- text 0.343 -> 0.323
- succ 0.972 -> 0.367 -> 0.228

## Mean ± sd over seeds 1-3 (range in brackets)

| | base | rc | rc+rep |
|---|---|---|---|
| bits total | 829,816 ± 1,531 [2,981] | **777,156 ± 589** [1,155] | 825,072 ± 866 [1,519] |
| text bits | 147,559 ± 1,630 | 144,951 ± 1,432 | 165,311 ± 589 |
| b P0 copy / c P0 | 0.977 / 0.725 | 0.996 / 0.817 | 0.995 / 0.825 |
| b P1 rev | 0.795 ± 0.002 | 0.952 ± 0.004 | 0.956 ± 0.002 |
| b P3 succ | 0.504 ± 0.003 | 0.720 ± 0.012 | 0.694 ± 0.005 |
| b P4 copy / P5 rev | 0.989 / 1.000 | 1.000 / 1.000 | 0.987 / 0.987 |
| text 17k -> 29k | 0.356 -> 0.243 ± 0.016 (68% kept) | 0.339 -> 0.244 ± 0.006 (72%) | 0.326 -> **0.299 ± 0.020** (92%) |
| succ 23k | 0.972 | 0.991 | 0.996 |
| succ 26k | 0.152 ± 0.014 | **0.950 ± 0.033** | 0.885 ± 0.050 |
| succ 29k | 0.045 ± 0.020 | **0.811 ± 0.019** [0.033] | 0.608 ± 0.087 [0.167] |
| copy / rev 29k | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Replayed episodes per phase:
- rc+rep seed 1 (log): succ phase copy 587, rev 713, text 523, succ 927; copy phase succ 541 of 750; final rev phase succ 593 of 750.
- Seed 2 and seed 3 logs show a similar make-up.
- Targeted replay moves most of its draws onto succ lines once copy and rev training raises their error.

## Verdict (plain)
1. **The seed-1 rule-cell result holds on seeds 2 and 3.** Over three seeds:
   - Succ at the end is 0.811 ± 0.019, against 0.045 for the baseline. After the copy phase it is 0.950, against 0.152.
   - Copy and rev are 1.00.
   - The life costs 777,156 ± 589 bits, against 829,816 (-6.3%).
   - First-met rev b is 0.95 against 0.80, succ b 0.72 against 0.50, and P0 copy c 0.82 against 0.725.
   - The spread over seeds is small: bits range 1,155, succ at the end 0.789-0.822.
   - As on seed 1, the text is not helped: 0.339 -> 0.244, 72% kept (baseline 68%).
2. **Replay adds text keeping, but it costs part of what the rule cells gave.**
   - Text: rule cells + targeted replay keep 92% of the text probe (0.326 -> 0.299, three seeds), against 72% with rule cells alone.
     On all three seeds the text at the end is 0.28-0.32, against 0.24-0.25.
   - Succ: the learned mapping is kept less well. At the end it is 0.608 ± 0.087 (0.71 / 0.58 / 0.54), against 0.811 ± 0.019 with
     rule cells alone. It falls during both the copy phase (0.885 at 26k) and the final rev phase.
   - Bits: the life costs 825,072 bits, +6.2% over rule cells alone and about the baseline's level. The extra cost is mostly in the
     text phase: text bits are 165k against 145k, as in IDEA12, where replayed rule lines interleaved with the text disturb its
     prediction. First-met succ b also drops a little (0.694 against 0.720).
   - Only seed 1 meets the condition set for continuing (succ >= 0.7 and text >= 0.30 at the end). Seeds 2 and 3 miss it on both:
     succ 0.58 / 0.54, text 0.298 / 0.280.
   - Against replay without rule cells (seed 1), rule cells + replay is much better on succ (0.71 against 0.23) and on bits
     (825,552 against 865,721), with the same text keeping (0.319 against 0.323).
3. So targeted replay of this store at 1 per 4 lines does **not** add text keeping for free on top of rule cells. It trades about
   0.2 of the learned mapping at the end, and all of the rule cells' bit saving, for about 0.055 of text probe.

## Limits / NOT_RUN
- cue=0 was not run.
- Only rho 0.25 and only targeted replay were run with rule cells. Random replay, rho 0.125 and replay of text episodes only were not
  run.
- No battery or natural-text tests were run with rc.
- One rule-cell setting was tested: rc=64, rc_gate=1, vigilance 0.4.
- Probes use 60 items per rule and 1,000 text bytes. The text-probe spread over seeds is about 0.02.
