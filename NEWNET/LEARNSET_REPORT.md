# LEARNSET report: does the network learn to learn? (23 November 2026)

This is a helper run on branch `claude/newnet-learnset-sei232`, from `claude/magical-albattani-sei232` @ 6df08ef8.
- Development level, one machine. **No gate, seal or outside review is claimed.**
- The plan, the rule list and the criterion were committed before any full run (cd7522a0, `LEARNSET_PLAN.md`).
- Code is in new files only: `t28_learnset.py` (the life), `t28_summary.py` (per-rule tables and slopes) and `t28_overlap.py`
  (reuse of position/lag pairs). No existing `.py` file was edited.
- Logs are in `learnset_runs/`: one `*.log` and `*.time` per life, all results in `res.jsonl`, full per-seed tables in `summary.md`,
  and the run script `run_s1.sh`.
- Learning is local (no backprop). No net option was retuned.

## What was run
**The life.** One life with nothing reset:
- 20 rules, 1,500 lines each, each rule never seen before. Lines are `w D W\n`, with w = 3 of 40 symbols and all 40 present from the start.
- Then a revisit block: rules 1, 5, 9, 13 and 17 again, 500 lines each.
- Then probes with learning off. Bits are counted as in t23.

**The rule kinds:**
- **P** (6): position permutations. P012 = copy, P210 = reverse.
- **M** (6): the same permutations followed by one fixed random symbol map SIGMA, shared by all M rules.
- **C** (8): a permutation with a constant symbol inserted at position q.

**The order:** C M P M C C M M C P P C P C M C M C P P (the rules are listed in the plan).

**The delimiter:**
- cue=1: each rule has its own new delimiter.
- cue=0: all rules share `=`.

**What is scored:** the life's own answer bytes for items met for the first time, prequentially, while learning.

**Lives:** 16, all finished.
- Seed 1: net12 (`lagro=1 relay=1`), net12_rc (`rc=64 rc_gate=1 lagro=1 relay=1`), K8 net10 and storage, each with cue 1 and 0.
- Seeds 2-3: the two net12 variants, cue 1 and 0.
- All net lives used the K8 options given in the task.

**Cost (measured):**
- A net life read about 264,000 bytes with learning on and took 22.7-26.4 min on one core; storage took 9 s.
- 14 net lives, 4 in parallel, took about 1 h 50 min wall clock. No checkpoints were kept.

## The learning curves over rules

**cue=1** (L80 in lines, '-' = not reached in 1,500; f20 / f100 / f500 right shares; net12 variants: mean of seeds 1-3)

| rule | name | kind | net12 L80 | net12 f20/f100/f500 | net12_rc L80 | net12_rc f20/f100/f500 | K8 L80 | K8 f20/f100/f500 | storage L80 | storage f20/f100/f500 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C102_q3k# | C | 376 | 0.05/0.24/0.59 | 147 | 0.03/0.46/0.86 | - | 0.00/0.10/0.15 | - | 0.00/0.07/0.13 |
| 2 | M210 | M | - | 0.02/0.02/0.09 | - | 0.01/0.02/0.13 | - | 0.00/0.03/0.20 | - | 0.12/0.22/0.36 |
| 3 | P210 | P | 772 | 0.02/0.05/0.24 | 623 | 0.04/0.08/0.39 | - | 0.00/0.05/0.20 | - | 0.08/0.22/0.36 |
| 4 | M012 | M | - | 0.01/0.03/0.05 | 1408* | 0.02/0.03/0.12 | - | 0.00/0.00/0.02 | - | 0.02/0.02/0.01 |
| 5 | C102_q1k5 | C | 559 | 0.23/0.43/0.63 | 358 | 0.24/0.52/0.73 | - | 0.04/0.12/0.25 | - | 0.03/0.16/0.27 |
| 6 | C201_q3kO | C | 443 | 0.03/0.07/0.53 | 168 | 0.10/0.42/0.83 | - | 0.04/0.14/0.33 | - | 0.07/0.16/0.24 |
| 7 | M120 | M | - | 0.01/0.01/0.02 | - | 0.02/0.02/0.05 | - | 0.00/0.02/0.03 | - | 0.02/0.02/0.09 |
| 8 | M021 | M | - | 0.03/0.04/0.11 | - | 0.06/0.10/0.19 | - | 0.02/0.02/0.05 | - | 0.02/0.01/0.05 |
| 9 | C120_q2kf | C | 654 | 0.11/0.25/0.46 | 244 | 0.13/0.38/0.76 | - | 0.03/0.04/0.07 | - | 0.00/0.01/0.10 |
| 10 | P201 | P | 507 | 0.19/0.26/0.52 | 180 | 0.03/0.31/0.81 | - | 0.00/0.03/0.15 | - | 0.05/0.22/0.32 |
| 11 | P012 | P | 969 | 0.02/0.04/0.23 | 162 | 0.01/0.41/0.83 | - | 0.02/0.02/0.03 | - | 0.02/0.01/0.01 |
| 12 | C021_q1k2 | C | 606 | 0.28/0.36/0.62 | 384 | 0.50/0.50/0.70 | - | 0.00/0.14/0.25 | - | 0.03/0.17/0.28 |
| 13 | P021 | P | 814 | 0.41/0.45/0.59 | 488 | 0.33/0.38/0.65 | - | 0.03/0.06/0.07 | - | 0.00/0.03/0.05 |
| 14 | C201_q0kQ | C | 610 | 0.11/0.29/0.47 | 225 | 0.14/0.46/0.80 | - | 0.12/0.31/0.42 | - | 0.30/0.41/0.49 |
| 15 | M201 | M | - | 0.02/0.02/0.03 | - | 0.02/0.03/0.12 | - | 0.02/0.01/0.09 | - | 0.05/0.22/0.31 |
| 16 | C102_q2k< | C | 633 | 0.07/0.13/0.41 | 184 | 0.11/0.41/0.80 | - | 0.04/0.07/0.14 | - | 0.00/0.01/0.07 |
| 17 | M102 | M | - | 0.03/0.02/0.02 | - | 0.02/0.02/0.10 | - | 0.00/0.01/0.03 | - | 0.00/0.02/0.06 |
| 18 | C102_q0k` | C | 608 | 0.15/0.36/0.52 | 437 | 0.16/0.38/0.64 | - | 0.12/0.25/0.28 | - | 0.25/0.27/0.29 |
| 19 | P102 | P | 411 | 0.46/0.48/0.69 | 322 | 0.21/0.47/0.72 | - | 0.05/0.06/0.08 | - | 0.02/0.02/0.05 |
| 20 | P120 | P | 522 | 0.64/0.65/0.71 | 123 | 0.49/0.72/0.92 | - | 0.08/0.10/0.15 | - | 0.02/0.02/0.09 |

**cue=0** (L80 in lines, '-' = not reached in 1,500; f20 / f100 / f500 right shares; net12 variants: mean of seeds 1-3)

| rule | name | kind | net12 L80 | net12 f20/f100/f500 | net12_rc L80 | net12_rc f20/f100/f500 | K8 L80 | K8 f20/f100/f500 | storage L80 | storage f20/f100/f500 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C102_q3k# | C | 367 | 0.05/0.23/0.60 | 148 | 0.04/0.45/0.86 | - | 0.00/0.10/0.15 | - | 0.00/0.07/0.13 |
| 2 | M210 | M | - | 0.02/0.02/0.03 | - | 0.02/0.02/0.05 | - | 0.03/0.04/0.17 | - | 0.05/0.06/0.18 |
| 3 | P210 | P | 1085 | 0.02/0.02/0.03 | 325 | 0.00/0.10/0.64 | - | 0.00/0.01/0.02 | - | 0.00/0.03/0.07 |
| 4 | M012 | M | - | 0.00/0.01/0.02 | - | 0.00/0.01/0.02 | - | 0.00/0.01/0.03 | - | 0.02/0.02/0.03 |
| 5 | C102_q1k5 | C | 625 | 0.29/0.45/0.56 | 276 | 0.34/0.58/0.83 | - | 0.12/0.22/0.26 | - | 0.07/0.09/0.14 |
| 6 | C201_q3kO | C | 638 | 0.02/0.04/0.38 | 180 | 0.02/0.22/0.81 | - | 0.20/0.22/0.34 | - | 0.06/0.06/0.07 |
| 7 | M120 | M | - | 0.01/0.02/0.02 | - | 0.02/0.03/0.03 | - | 0.02/0.02/0.04 | - | 0.00/0.02/0.06 |
| 8 | M021 | M | - | 0.16/0.15/0.22 | - | 0.35/0.32/0.37 | - | 0.27/0.22/0.25 | - | 0.13/0.12/0.14 |
| 9 | C120_q2kf | C | 726 | 0.06/0.20/0.40 | 252 | 0.05/0.22/0.74 | - | 0.03/0.09/0.17 | - | 0.07/0.11/0.15 |
| 10 | P201 | P | 702 | 0.08/0.12/0.33 | 316 | 0.03/0.03/0.60 | - | 0.22/0.23/0.29 | - | 0.05/0.09/0.12 |
| 11 | P012 | P | 1395 | 0.01/0.03/0.14 | 315 | 0.01/0.03/0.63 | - | 0.07/0.05/0.06 | - | 0.03/0.02/0.02 |
| 12 | C021_q1k2 | C | 560 | 0.31/0.39/0.64 | 304 | 0.41/0.44/0.74 | - | 0.03/0.19/0.27 | - | 0.04/0.03/0.07 |
| 13 | P021 | P | 749 | 0.43/0.45/0.58 | 287 | 0.48/0.55/0.78 | - | 0.08/0.07/0.08 | - | 0.05/0.09/0.09 |
| 14 | C201_q0kQ | C | 733 | 0.20/0.28/0.56 | 182 | 0.28/0.42/0.85 | - | 0.25/0.42/0.49 | - | 0.05/0.15/0.21 |
| 15 | M201 | M | - | 0.02/0.02/0.03 | - | 0.02/0.02/0.05 | - | 0.02/0.03/0.09 | - | 0.05/0.03/0.04 |
| 16 | C102_q2k< | C | 1009 | 0.04/0.07/0.28 | 299 | 0.06/0.24/0.70 | - | 0.04/0.10/0.18 | - | 0.15/0.10/0.11 |
| 17 | M102 | M | - | 0.03/0.02/0.02 | - | 0.03/0.02/0.04 | - | 0.00/0.02/0.03 | - | 0.02/0.02/0.04 |
| 18 | C102_q0k` | C | 619 | 0.23/0.44/0.63 | 288 | 0.04/0.26/0.72 | - | 0.10/0.26/0.29 | - | 0.01/0.04/0.06 |
| 19 | P102 | P | 437 | 0.29/0.37/0.64 | 437 | 0.11/0.22/0.60 | - | 0.02/0.05/0.10 | - | 0.17/0.21/0.18 |
| 20 | P120 | P | 946 | 0.68/0.68/0.69 | 307 | 0.48/0.54/0.78 | - | 0.40/0.37/0.39 | - | 0.22/0.24/0.29 |

`*` = not reached on some seeds; the mean is over the seeds that reached it. Per-seed values are in `learnset_runs/summary.md`.
Across seeds 1-3 the per-rule values agree closely (L80 typically within ±30 lines).

## Fitted trend
The fit is least squares over rule number, within each kind. Values are the mean ± sd over seeds 1-3 (K8 and storage: seed 1 only).
- f-values are the change over 20 rules.
- L80 is in lines per rule; "not reached" counts as 1,501.

| net | cue | P: f20 / f100 / f500 | P: L80 per rule | M: f100 / f500 | M: L80 reached | C: f100 / f500 | C: L80 per rule | bits total (answer) |
|---|---|---|---|---|---|---|---|---|
| net12 | 1 | +0.71±0.07 / +0.68±0.03 / +0.58±0.01 | **-18.9±1.2** | -0.00 / -0.07 | 0 of 18 | +0.04 / -0.15 | +13.0±0.5 | 1,159,963 ± 2,092 (285,131) |
| net12 | 0 | +0.66±0.07 / +0.69±0.02 / +0.80±0.02 | **-24.0±1.0** | -0.01 / -0.02 | 0 of 18 | +0.07 / -0.03 | +19.3±2.2 | 1,155,828 ± 2,459 (296,196) |
| net12_rc | 1 | +0.46±0.10 / +0.61±0.07 / +0.43±0.02 | **-18.6±1.7** | -0.00 / -0.01 | 2 of 18 | -0.07 / -0.15 | +8.1±1.7 | 1,127,585 ± 1,424 (229,825) |
| net12_rc | 0 | +0.43±0.07 / +0.45±0.01 / +0.09±0.02 | +3.1±1.7 | -0.02 / -0.02 | 0 of 18 | -0.19 / -0.15 | +6.2±0.6 | 1,080,272 ± 2,621 (223,554) |
| K8 net10 | 1 | +0.09 / +0.06 / -0.08 | never reached | -0.01 / -0.08 | 0 of 6 | +0.14 / +0.09 | never reached | 1,567,881 (580,349) |
| K8 net10 | 0 | +0.24 / +0.23 / +0.25 | never reached | -0.02 / -0.09 | 0 of 6 | +0.13 / +0.12 | never reached | 1,486,831 (543,149) |
| storage | 1 | -0.07 / -0.25 / -0.33 | never reached | -0.02 / -0.04 | 0 of 6 | +0.16 / +0.12 | never reached | 2,646,702 (978,899) |
| storage | 0 | +0.25 / +0.26 / +0.23 | never reached | -0.03 / -0.11 | 0 of 6 | -0.00 / -0.03 | never reached | 2,511,984 (1,078,613) |

### Revisits (500 lines; f100 first block -> revisit, seeds 1/2/3)
| net, cue | r1 C | r5 C | r9 C | r13 P021 | r17 M102 |
|---|---|---|---|---|---|
| net12, 1 | 0.23->0.84 / 0.22->0.83 / 0.26->0.81 | 0.42->0.99 / 0.43->0.97 / 0.44->0.99 | 0.28->0.89 / 0.25->0.87 / 0.23->0.90 | 0.46->0.59 / 0.48->0.57 / 0.41->0.56 | 0.02->0.02 |
| net12_rc, 1 | 0.42->0.69 / 0.45->0.73 / 0.50->0.58 | 0.50->0.98 / 0.55->0.95 / 0.52->0.97 | 0.40->0.93 / 0.40->0.90 / 0.34->0.97 | 0.39->0.60 / 0.38->0.69 / 0.38->0.67 | 0.02->0.02 |
| net12, 0 | 0.22->0.62 / 0.22->0.64 / 0.26->0.63 | 0.45->0.71 / 0.45->0.69 / 0.45->0.72 | 0.20->0.47 / 0.21->0.51 / 0.19->0.46 | 0.46->0.30 / 0.46->0.30 / 0.43->0.25 | 0.02->0.02 |
| net12_rc, 0 | 0.42->0.51 / 0.45->0.55 / 0.48->0.52 | 0.52->0.70 / 0.58->0.67 / 0.65->0.68 | 0.24->0.51 / 0.23->0.38 / 0.20->0.57 | 0.61->0.32 / 0.62->0.31 / 0.41->0.43 | 0.01-0.04 |

Keeping probes (learning off) after rule 20, mean over the rules of each kind (P / M / C):
- net12 cue=1: 0.72-0.74 / 0.02-0.03 / 0.77-0.78.
- net12_rc cue=1: 0.73-0.83 / 0.01-0.03 / 0.64-0.73.
- With cue=0 these are lower: 0.32-0.35 over all rules for both net12 variants.

## What the curves show
1. **P (position permutations): the start level of a new rule rises from rule to rule.**
   - This holds on all 3 seeds for net12 (cue 1 and 0) and net12_rc (cue 1).
   - With cue=1, net12's f100 goes 0.05 (rule 3, P210) -> 0.26 (rule 10) -> 0.46 (rule 13) -> 0.64 (rule 20).
   - L80 falls by about 19-24 lines a rule.
   - This rise is in **f20**, the first 20 lines of a rule whose delimiter has never been seen:
     - by rule 20 (P120), 0.60-0.70 of first-met answer bytes are right from the start;
     - the f20 slope (+0.71) is the same as the f100 slope (+0.68).
   - It is mostly a better starting guess, **not faster learning after the start**. In net12, P120 starts at 0.64 and still needs
     ~520 lines to reach 0.8; rule 3 needed ~770 lines from 0.05.
   - net12_rc (cue=1) does learn some later P rules faster from a chance start:
     - P201 (rule 10) and P012 (rule 11): f20 0.00-0.05, L80 140-210;
     - P210 (rule 3): L80 590-645.
     - But this is not monotone: P021 (rule 13) took 456-531 and P102 (rule 19) 292-344.
2. **The gain matches reuse of components that earlier rules taught** (`t28_overlap.py`).
   - Each rule is a set of (answer position, source lag) pairs. Every P rule from rule 10 on uses only pairs that earlier P/C rules
     already used; rule 3 used none.
   - P120 (rule 20) has the pairs used most often before: 5, 3 and 4 earlier uses.
   - The copy rule P012 (rule 11) had only 0, 1 and 1 earlier uses, and it is the outlier: net12 f100 0.04, L80 ~970.
   - In this design, rule number and pair reuse cannot be separated: every late P rule is built from familiar pairs, since there are
     only 9 such pairs in all.
   - Storage with cue=0 shows a smaller P rise from the same order (+0.26 f100), because contexts carry over the shared `=`.
     With cue=1 storage falls (-0.25). net12's +0.68 is well above both.
3. **M (permutation + the shared symbol map): no learning to learn, and no reuse of the map.**
   - No M rule reaches 0.8 in 1,500 lines, except M012 (rule 4) on net12_rc, 2 of 3 seeds (1,346-1,471 lines).
   - f100 stays 0.01-0.04 from the first M rule (2) to the last (17), even though SIGMA is the same in all six.
   - The direct readout learns SIGMA again per lag, or not at all. A shared mapping is not transferred.
   - This matches the note that the relay helps permutations, not symbol maps.
4. **C (constant insertion): no gain, and new C rules get slower.**
   - f500 falls by 0.15 over 20 rules for both net12 variants at cue=1.
   - L80 rises by 8-13 lines a rule (cue=1) and 6-19 (cue=0). This is interference from the earlier rules, the opposite of a
     learning set.
   - net12_rc learns C rules fast from rule 1 (L80 ~150), but not faster later on.
5. **cue=0 (shared delimiter):**
   - net12 shows the same P rise. Rule 20 starts at f20 0.68: the "previous rule plus pair prior" guess.
   - net12_rc's P rule L80 does **not** fall (+3.1 lines a rule; f500 slope +0.09), and its C rules get worse (f100 -0.19).
6. **K8 and storage reach 0.8 on no rule.** Their small slopes come from 1-gram/2-gram contexts carried over from earlier rules.
7. **Revisits:**
   - C rules revisited are re-learned at once (cue=1: f100 0.81-0.99 against 0.22-0.50 the first time). This is keeping (E2),
     not learning to learn.
   - P021 is partly kept (0.56-0.69).
   - M102 is not kept (0.02).
8. **Bits:**
   - net12_rc costs the fewest bits: about 32,000 fewer than net12 at cue=1 (-2.8%) and about 76,000 fewer at cue=0 (-6.5%).
   - Most of the saving is on answer bytes: 230k against 285k at cue=1.
   - K8 costs about 1.35x net12, and storage about 2.2x.

## Against the criterion fixed in the plan
The criterion: f100 and f500 slopes > 0, a fitted f100 change >= 0.10, L80 slope <= 0, and for the net12 variants all 3 seeds.

| | P | M | C | all 20 |
|---|---|---|---|---|
| net12 cue=1 | **met** | not met | not met (f500 and L80 worse) | met (carried by P; M flat, C worse) |
| net12 cue=0 | **met** | not met | not met | met (carried by P) |
| net12_rc cue=1 | **met** | not met | not met | met (carried by P) |
| net12_rc cue=0 | not met (L80 +3.1 a rule) | not met | not met | met on f100/f500/L80 by the mix; P alone fails |
| K8 cue=1 / 0 | not met (L80 never reached) | not met | not met | not met |
| storage cue=1 / 0 | not met | not met | not met | not met |

Predictions in the plan, against what happened:
- "P fast from rule 1": wrong. P210 at rule 3 took 600-1,100 lines.
- "M improves by reuse of SIGMA": wrong. There is no reuse.
- "K8/storage no L80 trend": right.
- "cue=0 curves get worse": wrong for P, right for C under net12_rc.

## Verdict
**Partly, and only for position permutations. Not in the strong sense of a general learning set.**
- In net12 and net12_rc, each new position-permutation rule starts higher and reaches criterion sooner than the earlier ones. This
  holds on all three seeds, and the formal criterion for P is met for net12 (cue 1 and 0) and net12_rc (cue 1).
- The gain is mostly in the first 20 lines. It follows how often the new rule's (position, lag) pairs were used by earlier rules: an
  accumulated prior over components, read out at once even for a never-seen delimiter.
- Learning after the start does not get uniformly faster.
- For symbol maps (M), nothing carries over, not even the map they all share.
- For constant-insertion rules (C), new rules get **slower** (interference).
- K8 net10 and storage do not learn to learn at all.
- This design cannot tell "learning to learn" from "reuse of the finite set of position/lag pairs". Every late P rule consists of
  familiar pairs.

## Limits / NOT_RUN
- K8 and storage were run on seed 1 only.
- One fixed rule order. Rule number and the particular rule are confounded within a kind, and there are only 6 P rules.
- No late rule with never-used (position, lag) pairs. That is the test that would separate a general learning set from component reuse.
  It needs a family with more pairs, for example longer words, and was not run.
- Probes are teacher-forced right shares, not whole own answers with EOS.
- No natural text in this life.
