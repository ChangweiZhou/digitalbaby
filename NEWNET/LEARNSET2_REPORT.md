# LEARNSET2 report: learning to learn, or reuse of familiar (position, lag) pairs? (23 November 2026)

Helper run on branch `claude/newnet-learnset2-sei232`, from `claude/magical-albattani-sei232` @ d0f0a9da.
- Development level, one machine. **No gate, seal or outside review is claimed.**
- The plan, the script and the predictions were committed before any counted run (78b03ddd, `LEARNSET2_PLAN.md`).
- Code is in new files only: `t29_learnset2.py` (the lives) and `t29_summary.py` (tables and the plan's classification). No existing
  `.py` file was edited. Learning is local (no backprop). No net option was retuned.
- Logs are in `learnset2_runs/`: one `*.log` and `*.time` per run, all results in `res.jsonl`, the full tables in `summary.md`, and the
  run script `run_all.sh`.
- Cost (measured): 56 runs, 4.94 core-hours. A history life (120,000 bytes) took 8.5-10.5 min on one core. Wall clock was
  14:52-16:07 UTC on 4 processes. No checkpoints were kept.

## What was run (as planned)
**Lines:** `w D W\n`, with 4-symbol words w from the 40-symbol alphabet of t23/t28. Each rule has its own delimiter.
- A position rule sets W[i] = w[f(i)].
- When W[i] is predicted, w[j] is lag 4+i-j back (1..7, within W=8).
- The pair of answer position i is therefore (i, 4+i-f(i)).

**Halves of the 16 pairs:**
- A has lags {3, 4, 7}; B has lags {1, 2, 5, 6}.
- History H_A or H_B is 8 non-constant selection rules of one half, and uses each pair of that half 4 times.

**Test rules:** the four rotations, whose lag sets are pairwise disjoint.
- X1 = copy (A), X2 = rot1 (A), Y1 = rot2 (B), Y2 = rot3 (B).

**The lives:**
- Life A: H_A, then Y1 (new), X1 (familiar), Y2 (new), X2 (familiar).
- Life B: H_B, then X1 (new), Y1 (familiar), X2 (new), Y2 (familiar).
- Lives RA and RB: 8 blocks of random answers (format only, no rule), then the test rules in the order of A or B.
- Fresh: each test rule as the first rule of a new net.

**Pair checks:** the script asserts the pair counts:
- every new-pair test rule had 0 prior uses of every one of its pairs;
- every familiar-pair test rule had 4 prior uses of each of its pairs.

**Blocks and nets:**
- Each block is 1,000 lines.
- net13 (rule cells, rc=64; equal to net12_rc): seeds 1-3.
- net12 `lagro=1 relay=1`: seeds 1-3.
- K8 net10: seed 1.
- All nets used the K8 options given in the task.

**Primary measure: err80**, the number of wrong first-met answer bytes before the right share over the last 20 lines first reaches 0.8.
- If that level is never reached, it counts every wrong byte in the block.
- The pilot showed that the t28 L80 (100-line window) floors at 100 on fresh nets.
- All scores are prequential with learning on, on the life's own teacher-forced answers.

## Results: net13 (rule cells), seeds 1-3 (values per seed)

| rule | fresh err80 | new err80 | fam err80 | rnd err80 | fresh / new / fam / rnd f100 (seed 1) |
|---|---|---|---|---|---|
| X1 copy | 62 / 47 / 50 | 239 / 166 / 217 | 151 / 105 / 166 | 70 / 57 / 66 | 0.82 / 0.47 / 0.59 / 0.73 |
| X2 rot1 | 472 / 459 / 514 | 1436 / 1520 / 1537 | 906 / 1024 / 925 | 1163 / 1110 / 1138 | 0.52 / 0.18 / 0.52 / 0.39 |
| Y1 rot2 | 230 / 211 / 232 | 708 / 660 / 677 | 285 / 244 / 245 | 618 / 467 / 537 | 0.46 / 0.17 / 0.47 / 0.19 |
| Y2 rot3 | 126 / 150 / 130 | 308 / 375 / 340 | 203 / 155 / 178 | 271 / 234 / 300 | 0.61 / 0.32 / 0.49 / 0.37 |

f20 is 0.00-0.08 for every new-pair block and 0.09-0.41 fresh.

**Lines to criterion, L80s (fresh -> new):**
- X1: 31-36 -> 63-128.
- Y1: 93-117 -> 267-317.
- Y2: 60-71 -> 135-182.
- X2: 386-424 -> not reached or 974-980. New-pair X2 stays at last500 0.72-0.74, against 0.98 fresh.

## Results: net12 (lagro=1 relay=1), seeds 1-3

| rule | fresh err80 | new err80 | fam err80 | rnd err80 |
|---|---|---|---|---|
| X1 copy | 526 / 516 / 542 | 3250 / 3266 / 3307 | 559 / 522 / 653 | 1281 / 1315 / 1332 |
| X2 rot1 | 1170 / 1143 / 1203 | 3787 / 3805 / 3823 | 1759 / 1764 / 1609 | 2391 / 2238 / 2270 |
| Y1 rot2 | 1194 / 1167 / 1139 | 3570 / 3618 / 3621 | 1349 / 1302 / 1361 | 2563 / 2573 / 2566 |
| Y2 rot3 | 986 / 924 / 883 | 3674 / 3666 / 3676 | 1087 / 922 / 1173 | 2170 / 2175 / 2158 |

For net12, **no new-pair block reaches even 0.5**, on any rule or seed: last500 is 0.07-0.35 and f500 is 0.02-0.03. The same rules
fresh reach 0.8 in 235-662 lines (last500 0.83-1.00).

## The comparisons fixed in the plan (12 paired cases per net: 4 rules x 3 seeds; same rule and seed)

| net | comparison | mean err80 (ratio) | a < b / a > b | mean f100 | f20 a - b | class |
|---|---|---|---|---|---|---|
| net13 | **new vs fresh** | 682 vs 224 (**3.05**) | 0 / 12 | 0.27 vs 0.61 | -0.18 | **slower** |
| net13 | fam vs fresh | 382 vs 224 (1.71) | 0 / 12 | 0.51 vs 0.61 | -0.16 | slower |
| net13 | new vs fam | 682 vs 382 (1.78) | 0 / 12 | 0.27 vs 0.51 | -0.03 | slower |
| net13 | new vs rnd | 682 vs 503 (1.36) | 0 / 12 | 0.27 vs 0.45 | -0.07 | slower |
| net13 | rnd vs fresh | 503 vs 224 (2.25) | 0 / 12 | 0.45 vs 0.61 | -0.11 | slower |
| net12 | **new vs fresh** | 3589 vs 949 (**3.78**) | 0 / 12 | 0.02 vs 0.15 | -0.11 | **slower** |
| net12 | fam vs fresh | 1172 vs 949 (1.23) | 1 / 11 | 0.34 vs 0.15 | +0.16 | no clear difference |
| net12 | new vs fam | 3589 vs 1172 (3.06) | 0 / 12 | 0.02 vs 0.34 | -0.27 | slower |
| net12 | new vs rnd | 3589 vs 2086 (1.72) | 0 / 12 | 0.02 vs 0.04 | -0.00 | slower |
| net12 | rnd vs fresh | 2086 vs 949 (2.20) | 0 / 12 | 0.04 vs 0.15 | -0.11 | slower |
| K8 (seed 1) | new / fam / rnd vs fresh | 3552 / 3520 / 3655 vs 3518 | - | 0.05 / 0.06 / 0.03 vs 0.09 | - | no clear difference (reaches 0.8 on no block) |

- Without X2: net13 new vs fresh 410 vs 138 (2.98), 0 of 9 faster; net12 3516 vs 875 (4.02), 0 of 9.
- Per seed (net13, mean err80 new / fam / rnd / fresh):
  - seed 1: 673 / 386 / 530 / 222;
  - seed 2: 680 / 382 / 467 / 217;
  - seed 3: 693 / 378 / 510 / 232.
- Per seed (net12): 3570-3607 / 1128-1199 / 2075-2101 / 938-969.
- The three seeds agree closely.
- "fam" and "rnd" blocks occupy different slots than the "new" blocks of the same rule. new and rnd share a slot; fam is at slot 10 or 12.

**The history rules themselves do not get faster** (err80 per slot 1-8):
- net13 A: 146-266 at slot 1, 1658-1722 at slot 3, 651-745 at slot 8.
- net12 A: 814-837 at slot 1, 1010-1101 at slot 8.
- The ups and downs are rule-specific and the same on every seed.

**Keeping (probes with learning off, after the history -> at the end; mean over the 8 history rules):**
- net13: A 0.72-0.78 -> 0.70-0.77; B 0.81-0.88 -> 0.80-0.83.
- net12: 0.80-0.84 -> 0.74-0.77.
- The new-pair test rules learned after the history are partly lost by the end of the life: net13 Y2 0.08-0.30 at the end, X2 0.17-0.50.

**Bits of the life:**
- net13: A 518,583-521,007, B 520,896-522,780.
- net12: 545,441-547,528.
- K8: 708,421 / 714,007.
- Lives with a random-answer history: 705-713k (most of that is the unpredictable answers).

## Against the predictions fixed in the plan
1. new vs fresh not faster: **right, and stronger than expected**. Both nets are "slower" on 12 of 12 pairs: 3.0x (net13) and 3.8x
   (net12) the errors of a fresh net. For net12 the new-pair rules are not learned at all within 1,000 lines.
2. fam vs fresh faster: **wrong**.
   - net13: familiar-pair rules are slower than fresh (1.71x, 12 of 12).
   - net12: "no clear difference" (1.23x). Only the f20 part holds (+0.16): familiar pairs give a head start in the first 20
     lines, then learning is slower than fresh.
3. new slower than fam in >= 9 of 12: **right**, 12 of 12 for both nets.
4. new vs rnd no clear difference: **wrong**. New-pair rules after the rule history are slower than after a random-answer history
   (1.36x net13, 1.72x net12; 12 of 12).
5. K8 uninformative: **right**. It reaches 0.8 on no block in any condition.

## Verdict
**No learning to learn.** Rules made of (position, lag) pairs the life had never used are learned **slower** after a history of
position rules than by a fresh net:
- net13 (rule cells): 3.0x the errors before criterion;
- net12: 3.8x, where they are not learned at all in 1,000 lines;
- 12 of 12 rule/seed pairs for each net.

The owner's criterion for learning to learn (new-pair rules faster after the history than in a fresh life) is not met for either net.
This answers the question LEARNSET left open: the earlier late-rule speed-up is component reuse, not a general learning set.

**What the history does instead:**
1. **A general cost of having lived:** after 8 blocks of random answers, the same rules cost 2.2x (both nets) the errors of a fresh net.
2. **A further cost specific to the rule history,** largest for rules whose lags were never relayed: new-pair rules cost 1.36x
   (net13) and 1.72x (net12) the errors after random answers.
3. **Familiar pairs give a head start, not faster learning:**
   - net12: f20 +0.16 over fresh, and familiar-pair rules cost 0.56x the errors of the same slot-type after random answers.
   - net13: familiar-pair rules are cheaper than after random answers (382 vs 503) but still dearer than fresh.

**Net comparison:**
- Rule cells (net13) soften every cost: new-pair 682 vs net12 3,589; fresh 224 vs 949; bits about 5% fewer.
- They do not turn any cost into a gain.

My reading, not tested: the gate and the readouts entrenched on the used lags suppress the unused lag lines. A fresh net starts with
no such bias. This makes plasticity of plasticity (the next step proposed in the memo) the open question, and gives it a measurable
target: new vs fresh err80 ratio <= 0.8 in this design.

## Limits / NOT_RUN
- One split of the pairs (halves with disjoint lags), so "new pairs" here also means "never-relayed lags". The split with shared lags
  but new pairs was NOT_RUN. In that split a lag-level reuse could help.
- One history order and 8 history rules. The history rules are selection maps with repeated symbols; the test rules are rotations.
- fam and rnd blocks of a rule sit at different slots (10/12 vs 9/11, and rnd shares the new block's slot). The ratios are large and
  consistent, but slot is not fully balanced for fam.
- K8 seed 1 only. Probes are teacher-forced right shares, not whole own answers with EOS. No natural text in this life.
- The pilot (seed 9, BL=400, not counted) showed X2 fresh plateauing at 0.75 within 400 lines. With 1,000 lines, X2 fresh reaches
  0.98 (net13) and 0.83 (net12). Comparisons are also given without X2; the verdict does not change.
