# LEARNSET2 plan: learning to learn, or reuse of familiar (position, lag) pairs? (23 November 2026)

Branch `claude/newnet-learnset2-sei232` from `claude/magical-albattani-sei232` @ d0f0a9da. Helper run, development level, one
machine (4 cores). **No gate, seal or outside review is claimed.** This plan, the script `t29_learnset2.py` and the predictions are
committed **before** any counted run. No existing `.py` file is edited. Learning is local (no backprop); no net option is retuned.

## Question (one hypothesis, one falsifier)
LEARNSET_REPORT.md found that later position-permutation rules start higher, but every late rule consisted of (answer position,
source lag) pairs that earlier rules had used, so learning to learn could not be told from component reuse.
**Hypothesis H (learning to learn):** after a history of position rules, a new rule built only from (position, lag) pairs, and here
even lags, that the life never used is learned faster than the same rule as the first rule of a fresh life.
**Falsifier:** new-pair rules after the history are not faster than in a fresh life (or slower), while familiar-pair rules are faster.

## Design
Lines `w D W\n` (no spaces), w = 4 symbols of the 40-symbol alphabet of t23/t28, D = the rule's own delimiter (cue=1), W = 4 answer
bytes. A position rule f: W[i] = w[f(i)]. When W[i] is predicted the last byte read is position 4+i, so w[j] is **lag 4+i-j** (1..7,
inside the W=8 window; rot1's last answer byte comes from 7 back) and the delimiter is at lag i. Pair of position i: (i, 4+i-f(i)).

The 16 pairs are split into two halves, each with 2 sources per answer position:
- **A**: f(i) = i + c_i (mod 4), c_i in {0, 1}: lags {3, 4, 7}.
- **B**: c_i in {2, 3}: lags {1, 2, 5, 6}.
A rule of a half is a 4-bit pattern (c_i = offset[bit_i]); its complement shares no pair with it.

**Test rules** (the four cyclic rotations; their lag sets are pairwise disjoint):
X1 = A0000 copy `w0w1w2w3`, X2 = A1111 rot1 `w1w2w3w0`, Y1 = B0000 rot2 `w2w3w0w1`, Y2 = B1111 rot3 `w3w0w1w2`.

**History**: the same 8 non-constant bit patterns (4 complementary pairs, fixed random order, rng 29) in half A (H_A) or B (H_B):
each pair of the half is used exactly 4 times; none of the other half. (`python3 t29_learnset2.py list=1` prints every rule, its
delimiter, its pairs and their prior uses; the script asserts that new-pair test rules have 0 prior uses of every pair and
familiar-pair rules >= 1.)

| life | blocks 1-8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|
| A | H_A | Y1 **new** | X1 fam | Y2 **new** | X2 fam |
| B | H_B | X1 **new** | Y1 fam | X2 **new** | Y2 fam |
| RA | 8 blocks of random answers (format only) | Y1 rnd | X1 rnd | Y2 rnd | X2 rnd |
| RB | the same | X1 rnd | Y1 rnd | X2 rnd | Y2 rnd |
| fresh | - | one test rule as the first rule of a new net | | | |

So every test rule is measured as a **new-pair** rule (one life), a **familiar-pair** rule (the other life), after a **random-answer
history** of the same length (same slot as its new-pair measure), and **fresh**. Each block is BL = 1,000 lines (10 bytes a line); a
history life reads 120,000 bytes. Delimiters: history/random slot k -> DH[k] in every life; test rule -> DT[rule] in every life.
Probes (learning off, 30 fresh items per rule) after the history and at the end; bits of the life as t28.

**Scores per block** (first-met answer bytes, prequential, learning on, own answers teacher-forced, as t28): f20 / f100 / f500, last500,
50-line bins, L80 / L50 (100-line window, as t28; floor 100), and, because the pilot below shows fresh nets reach 0.8 well within 100
lines, **L80s** (the same with a 20-line window) and **err80** = wrong first-met answer bytes before L80s (whole block if not reached).
**Primary measure: err80** (lower = faster; it has no floor and counts censored blocks fully).

**Nets** (no retuning; K8 options L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75):
- net13 (rule cells; `net=13`, defaults rc=64 = net12_rc), seeds 1-3;
- net12 `lagro=1 relay=1`, seeds 1-3;
- K8 net10, seed 1.
Runs: lives A, B, RA, RB for every net/seed (28 lives) and fresh X1, X2, Y1, Y2 (28 short lives). OMP_NUM_THREADS=1, 4 processes.

**Cost estimate**: ~4 ms a byte on one core (pilot) -> a 120,000-byte life ~8-10 min; all runs ~5 core-hours, ~75-90 min wall clock.
No checkpoints are kept (the nets are rebuilt from seed; logs and res.jsonl are kept).

**Pilot run before this plan** (seed 9, net13, fresh, BL=400; not counted, only used to choose err80/L80s and BL):
L80s / err80 / f100: X1 32 / 51 / 0.87; Y2 78 / 138 / 0.64; Y1 109 / 214 / 0.48; X2 not reached (plateau 0.75: 3 of 4 positions) / 454 / 0.62.

## Criteria (fixed now)
For each of net13 and net12 there are 12 paired cases per comparison (4 test rules x 3 seeds; pairs share rule and seed):
- **new vs fresh** (the owner's definition of learning to learn).
  - **Faster (learning to learn)**: mean err80(new) <= 0.8 x mean err80(fresh), new < fresh in >= 9 of 12 pairs, and mean f100(new)
    >= mean f100(fresh).
  - **Slower (interference)**: mean err80(new) >= 1.25 x mean err80(fresh) and new > fresh in >= 9 of 12.
  - Otherwise **no clear difference**.
- **fam vs fresh** (benefit of reuse), **new vs fam** (same rule and seed in the two lives), **new vs rnd** (format/warm-up control):
  the same three-way classification.
- Learning to learn **beyond format exposure** needs new vs fresh "faster" **and** new vs rnd "faster". If new beats fresh but not rnd,
  the gain is called warm-up/format exposure, not learning to learn from the rules.
- K8 (seed 1): 4 pairs per comparison, descriptive only.
- Also reported (descriptive): the err80 / f20 curve over the 8 history rules (within-half reuse), keeping probes, bits.

## Predictions (written before running)
1. net13 and net12, new vs fresh: **not faster**; I expect "slower" or "no clear difference" (the history's gate and readout favour
   the old lags; LEARNSET showed C rules getting slower).
2. fam vs fresh: faster (reuse), at least in f20 (mean f20(fam) - f20(fresh) >= +0.15) for both nets.
3. new vs fam: new slower in >= 9 of 12 pairs for both nets.
4. new vs rnd: no clear difference.
5. K8: reaches L80s on few or no test blocks in any condition; uninformative for the question.
If prediction 1 fails in the "faster" direction and the new vs rnd test is also "faster", hypothesis H is supported for that net.

## Limits known in advance / NOT_RUN
- One split of the pairs (halves with disjoint lags), one history order, 8 history rules; history rules are non-permutation selection
  maps, test rules are rotations. Second split with shared lags but new pairs: NOT_RUN.
- Probes are teacher-forced right shares, not whole own answers with EOS. No natural text in this life. Storage baseline not run
  (first-met bytes are at chance for it by construction).
