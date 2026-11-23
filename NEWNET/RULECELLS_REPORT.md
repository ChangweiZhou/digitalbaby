# Rule cells and attention-gated plasticity of the relay gate (23 November 2026)

Helper branch `claude/newnet-rulecells-sei232` (from `claude/magical-albattani-sei232` @ 9dcc213e). Development level, one machine,
**seed 1 only, cue=1 only**. No gate, seal or outside review is claimed. The plan and criteria were written before the runs
(`RULECELLS_PLAN.md`).
Code (new files only): `net12_rc.py` (subclass of net12.Tissue: `rc=` rule cells, `ag=1` attention-gated gate plasticity),
`toys/vig_toy_rc.py` (toys/vig_toy_lesion.py importing net12_rc, plus a count of which rule cells win on each rule's answer bytes),
and `t23_life_rules_rc.py` (t23_life_rules.py with net=12 importing net12_rc). Logs: `rc_runs/` (toy_*.log, full_*.log,
full_res.jsonl, run_toy.sh, run_full.sh). Learning stays local (no backprop). Toy: about 2 min per run. Full size: about 27 min per
run, 4 in parallel.

## Mechanisms as run
- **A, rule cells** (rc=64 by default; rc_k=3, rc_tau=8, rc_vig=0.4, rc_eta=0.02):
  - Each rule cell has a prototype over the gate's inputs (the lag cells, plus the match cells when match>0).
  - At each byte, the best-matching cell (cosine) ignites. If the best match is below rc_vig, the least-used cell is recruited
    (prototype = present inputs).
  - Ignited cells persist, decaying with tau=8 bytes (about one line). The k most active stay on.
  - Only the winner's prototype learns: Hebbian with competition, rate 1/uses, floored at rc_eta.
  - The gate reads the cells' normalised activity through strengths Gr, learned by the delta rule as net12's gate. rc_gate=1 adds
    this to the lag-cell gate; rc_gate=2 uses it instead of the lag-cell gate.
  - No byte is treated specially.
- **B, attention-gated plasticity** (ag=1, ag_tau=2000, ag_pow=1):
  - Each gate synapse keeps a running mean m and mean square v of its lag's error, updated while its input is active.
  - Its change is scaled by p = ((|m| + 0.01) / (sqrt(v) + 0.01)) ** ag_pow.
- **Check:** ag=1 with ag_pow=0 (p = 1) reproduces the match=4 baseline exactly, in every phase and every lesion (toy_chk_agpow0.log).

## Toy (4 regions; copy 2000 -> rev 2000 -> succ 3000 -> copy 1000 lines, cued; probes with learning off)
- "copy first200" = right share in the first 200 lines of the first copy phase.
- Last column = probes after the final copy (copy / rev / **succ**).
- Shared winners = rule cells that win on both copy and succ answer bytes in 60 probe lines each.

| run | copy first200 | after succ: copy / rev / succ | after final copy: c / r / **s** | relay lesioned: s | shared winners copy&succ | pass? |
|---|---|---|---|---|---|---|
| baseline net12 (earlier) | 0.73 | 0.53 / 0.97 / 1.00 | 1.00 / 1.00 / **0.58** | 0.97 | - | no |
| baseline match=4 | 0.70 | 0.72 / 0.97 / 1.00 | 1.00 / 1.00 / **0.81** | 0.92 | - | no |
| **A rc_gate=1** | 0.94 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / **0.98** | 0.96 | 0 | **yes** |
| A rc_gate=2 | 0.89 | 1.00 / 0.88 / 0.99 | 1.00 / 0.95 / **0.96** | 0.97 | 0 | yes (rev at the bar) |
| A rc_gate=1, match=4 | 0.93 | 0.99 / 0.89 / 1.00 | 1.00 / 0.98 / **0.96** | 0.97 | 1 | yes |
| A rc_gate=2, match=4 | 0.90 | 0.99 / 0.68 / 1.00 | 1.00 / 0.76 / **0.95** | 0.92 | 1 | no (rev) |
| B | 0.71 | 0.70 / 0.98 / 0.97 | 1.00 / 1.00 / **0.75** | 0.96 | - | no |
| B, match=4 | 0.63 | 0.84 / 0.99 / 0.99 | 1.00 / 0.99 / **0.81** | 0.91 | - | no |
| B ag_pow=2 | 0.71 | 0.76 / 0.98 / 0.97 | 1.00 / 1.00 / **0.67** | 0.96 | - | no |
| B ag_tau=10000 | 0.71 | 0.72 / 0.97 / 0.98 | 1.00 / 1.00 / **0.76** | 0.96 | - | no |
| A+B (rc_gate=1) | 0.94 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / **0.97** | 0.98 | 0 | yes |
| A+B, match=4 | 0.93 | 0.99 / 0.83 / 1.00 | 1.00 / 0.95 / **0.96** | 0.97 | 1 | yes (rev at the bar) |
| A rc_k=1 | 0.94 | 1.00 / 0.90 / 1.00 | 1.00 / 0.93 / **0.99** | 0.99 | 0 | no (rev 0.93) |
| A rc_k=6 | 0.93 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / **0.97** | 0.97 | 0 | yes |
| A rc_tau=1 (little persistence) | 0.95 | 1.00 / 0.91 / 1.00 | 1.00 / 0.97 / **0.99** | 0.98 | 0 | yes |
| A rc_vig=0.3 | 0.98 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / **0.98** | 0.99 | 0 | yes |
| A rc_vig=0.5 | 0.97 | 0.88 / 0.99 / 0.66 | 1.00 / 0.99 / **0.09** | 0.48 | 23 | no |
| A rc=16 cells | 0.98 | 0.54 / 0.99 / 0.64 | 1.00 / 0.91 / **0.19** | 0.31 | 7 | no |
| A rc_sur=0.5 (surprise reset) | 0.94 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / **0.99** | 0.98 | 0 | yes |
| A2 rc_k=1 | 0.90 | 1.00 / 0.89 / 1.00 | 1.00 / 0.89 / **0.98** | 0.99 | 0 | no (rev) |
| A2 rc_tau=1 | 0.91 | 1.00 / 0.86 / 1.00 | 1.00 / 0.89 / **0.98** | 0.98 | 0 | no (rev) |

Recruitments (toy, rc=64):
- vig 0.4: 22,931. Most are on the random input bytes; answer-byte winners are copy 14, rev 15, succ 7 cells.
- vig 0.3: 764.
- vig 0.5: 59,879. The cells churn, and copy and succ share 23 winners.
- 16 cells: 45,224.

## Full size (t23 life, K8 net12 lagro=1 relay=1, cue=1, seed 1)

| run | bits total (text / answer) | text probe after text -> end | succ after succ -> after copy -> end | copy / rev end | first-met P0 copy b / c, P1 rev b, P3 succ b |
|---|---|---|---|---|---|
| baseline (LIFE_RULES_REPORT.md) | 828,527 (146,104 / 104,926) | 0.35 -> 0.24 | 0.98 -> 0.17 -> **0.07** | 1.00 / 1.00 | 0.98 / 0.73, 0.80, 0.51 |
| **A rc=64 (rc_gate=1)** | **777,024** (143,714 / 68,398) | 0.35 -> 0.24 | 1.00 -> 0.98 -> **0.82** | 1.00 / 1.00 | 0.995 / 0.83, 0.95, 0.72 |
| A rc_vig=0.3 | 776,405 (143,421 / 68,692) | 0.36 -> 0.26 | 0.99 -> 0.98 -> **0.82** | 1.00 / 1.00 | 0.996 / 0.83, 0.95, 0.74 |
| A rc_gate=2 | 784,318 (143,660 / 71,459) | 0.34 -> 0.25 | 1.00 -> 0.97 -> **0.83** | 1.00 / 1.00 | 0.994 / 0.83, 0.93, 0.71 |
| A+B | 779,064 (143,673 / 69,127) | 0.35 -> 0.24 | 1.00 -> 0.98 -> **0.82** | 1.00 / 1.00 | 0.994 / 0.83, 0.94, 0.72 |

- "after succ" is the probe at line 23,000, "after copy" at line 26,000, and "end" at line 29,000.
- In run A, succ falls during the final rev phase: 0.90 at 27,000 and 0.82 at 28,000. It does not fall during the copy phase.
- Copy and rev return right at once: first window 1.00.

Against the full-size criterion (succ at end >= 0.5, copy/rev >= 0.95, bits within 2% of 828,527):
- **Met** by all four runs: succ 0.82-0.83, copy/rev 1.00.
- Bits are 5.3-6.2% **fewer** than the baseline. Most of the saving is on answer bytes: 68-71k against 105k.

## What it shows (seed 1, cue=1)
1. **Prediction confirmed.**
   - "+" lines win on rule cells that copy lines never win on: 0 cells shared by copy and succ answer bytes in every passing run.
   - So copy training cannot reopen the gate there. On the toy, succ after the final copy rises from 0.58 to 0.98. At full size it
     rises from 0.07 to 0.82 at the end; after the copy phase it is 0.98 against 0.17.
   - The rule cells also make rules faster to learn: first-200 copy 0.94 against 0.73; first-met rev b 0.95 against 0.80; succ b
     0.72 against 0.51.
2. **Persistence is not what does it in these tests.** With tau=1 (little persistence) or k=1 (only the current winner), succ is
   kept just as well (0.98-0.99); only rev suffers somewhat (0.89-0.97). What matters is competitive recruitment of
   context-specific cells. The prediction fails when that breaks down:
   - With too few cells (16) or too high a vigilance (0.5), the cells churn.
   - Copy and succ then share cells, and succ falls to 0.09-0.19.
3. **B fails.** Attention-gated plasticity alone does not meet the criterion (0.75; 0.81 with match=4, the same as the match=4
   baseline). Added to A it changes nothing.
4. **What remains:**
   - Succ still falls 0.98 -> 0.82 during the last 3,000 rev lines at full size.
   - The text loss is untouched: 0.35 -> 0.24, as in the baseline.

## Limits / NOT_RUN
- Seed 1 only.
- cue=0 not run. There all rules share "=", the rule cells could separate only by the answers already given, and this mechanism
  probably cannot separate rules there.
- Full size was not run with match=4. No other seeds, and no other battery tests (natural text, keeping tests) with rc.
- The rule cells separate on the delimiter at a lag. That is a cue present in the stream, not an inferred rule.

## Verdict
Idea A (rule cells read by the relay gate, added to the lag cells) passes the toy criterion and the full-size criterion at seed 1,
cue=1:
- Toy: succ 0.98 after the final copy; copy/rev 1.00; first-200 copy 0.94.
- Full size: succ at the end 0.82 against 0.07; copy/rev 1.00; 777,024 bits against 828,527 (-6.2%).

The persistence and decay specified for it are not needed in these tests; competitive context cells are what does it. Idea B
(attention-gated plasticity) fails the toy (0.75) and adds nothing to A.
