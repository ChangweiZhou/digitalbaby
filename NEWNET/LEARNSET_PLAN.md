# LEARNSET plan: does the network learn to learn? (written 23 Nov 2026, before any full run)

Helper branch `claude/newnet-learnset-sei232`, from `claude/magical-albattani-sei232` @ 6df08ef8. Development level, one machine;
**no gate, seal or outside review is claimed**. Owner's question: in one life, with a series of rules each never seen before, do the
lines needed to learn a new rule fall from rule to rule (a learning set, Harlow 1949)?
Script: `t28_learnset.py` (new; no existing file edited). Rule list: `python3 t28_learnset.py list=1 [cue=0]`. Logs: `learnset_runs/`.

## The life (one continuous life, nothing reset)
- Lines `w D W\n` (no spaces, as in t23). w = 3 symbols of the 40-symbol alphabet of t23, **all 40 from the start** (64,000 words,
  so nearly every item is met for the first time); D = the rule's delimiter; W = the answer.
- 20 rules, 1,500 lines each (30,000 lines), then a **revisit block**: life rules 1, 5, 9, 13, 17 again, 500 lines each (2,500 lines).
- Rule family (fixed by `default_rng(28)`, the same for all seeds and both cue settings):
  - **P** (6): W = w permuted (W[i] = w[pi[i]]), all 6 permutations of 3 positions (P012 = copy, P210 = reverse).
  - **M** (6): W[i] = SIGMA(w[pi[i]]): the same 6 permutations combined with **one fixed random derangement SIGMA** of the 40 symbols
    (shared by all M rules: after the first M rule, SIGMA is reusable; an M rule is still new as a whole).
  - **C** (8): w permuted by pi with a constant symbol k (one of the 40) inserted at position q (4 answer bytes); q = 0..3, two
    permutations each.
- Order of the life (rule number: name, cue=1 delimiter):
  1 C102_q3k# `-` · 2 M210 `R` · 3 P210 `]` · 4 M012 `m` · 5 C102_q1k5 `k` · 6 C201_q3kO `\` · 7 M120 `N` · 8 M021 `n` ·
  9 C120_q2kf `w` · 10 P201 `}` · 11 P012 `y` · 12 C021_q1k2 `7` · 13 P021 `j` · 14 C201_q0kQ `B` · 15 M201 `0` · 16 C102_q2k< `A` ·
  17 M102 `6` · 18 C102_q0k`` ` `` `~` · 19 P102 `J` · 20 P120 `)`.
  Kinds per position: P at 3, 10, 11, 13, 19, 20; M at 2, 4, 7, 8, 15, 17; C at 1, 5, 6, 9, 12, 14, 16, 18.
- **cue=1**: each rule its own new delimiter (20 bytes not among the 40 symbols and not `=`). **cue=0**: all rules use `=`; the rule
  change can then be found only from the answers.

## Measures (prequential, learning on; the life's own teacher-forced answer bytes)
Per rule block, on answer bytes of first-met items (rule, w):
- **L80** = lines into the block until the right share over the last 100 lines first reaches 0.8 (window end; >= 100; "not
  reached" if never); **L50** likewise at 0.5 (secondary, for rules that never reach 0.8).
- **f100**, **f500** = right share in the block's first 100 / 500 lines (always defined); f20 (resolution of very fast learning);
  last500; 50-line bins.
- Revisit blocks: the same measures; savings = revisit against the rule's first block.
- Probes with learning off after rule 20 and at the end: 30 fresh items of each rule (keeping, E2).
- Bits of the whole life (as t23) and its parts (input / answer bytes).

## Learning-to-learn criterion (fixed now)
For each net, cue and kind (P, M, C separately, and all 20), fit by least squares: metric = a + b * (rule number), for f100, f500 and
L80 (not reached counted as 1,501, and the number not reached stated).
- **Learns to learn** (for that kind): f100 slope > 0 **and** f500 slope > 0 with the first-to-last fitted change of f100 >= +0.10,
  and L80 slope <= 0; for the net12 variants the f100 slope must be > 0 on all 3 seeds.
- **Does not**: slope of f100 <= 0 or within the seed spread of 0, or the first-to-last fitted change < 0.10.
- The M kind can improve by reuse of SIGMA (transfer of a shared mapping) rather than a general learning set; reported as such.
- Expected (my prediction, written before running): the relay helps P and C (position permutations), not M; for net12 / net12_rc,
  P rules are learned fast from rule 1 on (little room to improve: f100 high already); M rules improve after the first M rule because
  SIGMA is reused by the direct readout - that is reuse, not learning to learn; K8 and storage show no downward L80 trend. With cue=0
  all nets are slower and interference grows with each rule (curves may get worse, the opposite of a learning set).

## Nets (no retuning; settings as given)
- `net=12 lagro=1 relay=1` (baseline net12), K8 options (L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9
  c_th_match=0.75).
- `net=12rc rc=64 rc_gate=1 lagro=1 relay=1` (net12_rc, settings of RULECELLS_REPORT.md), K8 options.
- `net=10` (K8 net10), K8 options. `net=store` (back-off table, orders 1-8).
- Seed 1, cue 1 and 0, all four nets (8 lives); then seeds 2-3 for the two net12 variants, cue 1 and 0 (8 lives).
  OMP_NUM_THREADS=1, up to 4 processes (nproc = 4).

## Cost (estimate before running; measured cost goes in the report)
32,500 lines x ~8.1 bytes = ~264,000 bytes a life read with learning on, plus 2 probes x 20 rules x 30 items (~10,000 bytes, learning
off). About 5 ms a byte for net12 on one core: ~25-40 min a life; 14 net lives at 4 in parallel: ~2.5 h. No checkpoints are stored.

## NOT_RUN (planned now)
- K8 and storage on seeds 2-3; other rule orders (the order is one fixed draw, so rule number and the particular rule are confounded
  within a kind); returns with more than 500 lines; natural text in the life.
