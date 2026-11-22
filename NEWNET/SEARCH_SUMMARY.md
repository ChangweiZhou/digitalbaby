# Architecture search by fast iteration: where it stopped (22 November 2026)

Owner, this session: search by fast iteration for a general architecture, "直到改进几乎停滞，我们找到一些最佳结构在小模型上无法区分优劣
为止". Development level; one machine and cloud helpers of the same kind; no gate, seal or outside review is claimed. Every result is in
`search_results*.tsv`, `ARCH_GENERAL.md` and the batch reports `SEARCH_B3/B4/B10/B11/B13_REPORT.md`.

## The battery (the same for every candidate; sizes kept about equal)

Moby Dick and the repository's Python code (both never used in design): read 16,000 bytes, then the next byte on 10,000 never-read bytes,
learning off. Keeping: A (a context stream on letters text also uses) through 16,000 bytes of text, at gaps G = 8, 4 and 2. Memory: the
byte after a gap of 16, 40,000 bytes. Score = the mean of the six.

## The stop

About 70 candidates in 13 batches. The last three rounds (batches 12 and 13 and two N1 combinations: 13 candidates, 4 of them seed
repeats) all scored 0.794-0.801, within the spread between seeds; nothing beat the leaders. The best structures cannot be told apart:

| structure | options | seeds | mean score | Moby Dick | code | keeping G=8/4/2, memory |
|---|---|---|---|---|---|---|
| **K8: two pathways** - a pure chain (8 regions) beside a random graph without loops (8 regions, one source each), joined at the output | `L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75` | 1-3 | 0.798 | 0.393 | 0.397 | all passed, every seed |
| **N1: a chain with shortcuts** - each region reads the one before and one random earlier region; 64 columns of 32 cells | `L=8 cC=64 M=32 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4` | 1-3 | 0.798 | 0.390 | 0.396 | all passed, every seed |
| variants of K8 (wider holder expansion, other rates, more growth, smaller holders) and of N1 (more growth) | Q6, Q8, Q3, Q7, R2 | 1 | 0.799-0.801 | 0.390-0.399 | 0.401-0.408 | all passed |

For reference (three seeds): graphs without loops read new streams best (0.395 / 0.403) but never keep the G=8 context (0.47-0.54);
the plain chain keeps but reads new streams worse (0.38).

## What the search established (each held across many candidates)

1. **One shared mismatch** over all output compartments, with one step shared by all, beats compartments that learn alone, everywhere.
2. **Holding cells carry memory across a gap** (without them memory falls to chance), but do not protect it: through later text their
   reading is rewritten (their states inside a context and in text differ too little).
3. **A long context is separated only by a path of context cells as long as the context.** A fixed depth keeps no context longer than
   its pure serial path; mixing a short context into a region's input undoes the separation, unless the codes are sparse.
4. **Sparse context codes** (4 winning columns of 64, not 8 of 128) give both separation and generalisation, and gain about twice as
   much from more reading (64,000 bytes: 0.425-0.429 against 0.400).
5. **Fewer columns with more cells each** (64 x 32 rather than 128 x 16 or 256 x 8) is better at equal size; 32 columns are too few.
6. **Paths of many lengths** (random graphs) generalise better than one long chain; random loops are harmful; structured feedback
   (each region reading the one below and the one above) is not.
7. Without effect: a sheet with distance-dependent wiring; holders reading the network; replay, synaptic cascades and slow-context codes
   for the two-text readout; slower or faster rates within a factor of two.

## What was learned and what was set

- Learned by the system: every output strength (delta rule, or yes/no); the context cells' connections (made to the cells that were
  active, strengthened, recycled).
- Set at birth, general: kinds of cells (context, holding with clocks of many lengths, sparse expansion), inhibition (k winners),
  adaptation, gain by the sender's variance, the wiring rule.
- Chosen by this search on its battery: the topology, the sparseness, the thresholds, the sizes. These are the search's findings,
  not learned in a life.

## Limits

One small size (about 16,000 context cells), short lives (16,000-64,000 bytes), two never-used streams; keeping tested with one kind
of context; never-shown combinations (generalisation proper) not tested; the two-text readout drift unresolved; yes/no feedback not
run on these structures; most single-seed rows not repeated.
