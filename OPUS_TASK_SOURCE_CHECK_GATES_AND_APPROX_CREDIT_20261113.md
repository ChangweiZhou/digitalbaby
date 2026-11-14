# Source check: approximate credit, undecided gates and composition (13 November 2026)

Read-only literature job with web access. Trains nothing. Do not put any person's name, this repository's name or project details into a search or a request to an outside service; search by paper titles, authors and technical terms only.

Write the result to `PROJECT_DOCUMENTS/LIT_CHECK_20261113/SOURCE_CHECK_GATES_AND_APPROX_CREDIT.md` on your own branch.

## Two points to settle first

The outside review of 13 November found the earlier audit of learned feedback in recurrent networks unreliable (`PROJECT_DOCUMENTS/LIT_CHECK_20261113/AUDIT_OF_BOTH_SOURCE_CHECKS.md`).

1. The study cited there as "no gain over random feedback" uses a Kolen–Pollack rule. Open its supplement and read the bar values of every arm in the figure that compares it with random feedback. Give the numbers, or say they cannot be read.
2. Marschall et al. on local online learning in recurrent networks: state its result for each rule and task, with the table or figure.
3. List every study that learns the backward weights of a recurrent network by a local rule. The audit found three; the review says at least five exist.

## The review itself

```
I need a source-checked literature review on one specific question. Open the
primary papers, including figures, tables and supplements; do not rely on
abstracts or secondary summaries. For every claim give the URL, the exact
sentence or table/figure it rests on, the network type (rate or spiking,
gated or not), the task, the number of seeds, and whether it is peer reviewed.
"Not found" is an acceptable answer; say what you searched.

Setting. A 256-unit GRU learns byte-level modular arithmetic with distractors
and two-step compositions answered from memory. Two training rules are
compared on identical data: (1) backpropagation through time; (2) the same
backward pass with every weight matrix replaced by its signs times one mean
magnitude per matrix (sign symmetry applied through time).

Observation. Both rules reach about 98% on single operations. On untaught
operator pairs rule 2 scores 37-54% against 63-89%. Its errors are near
misses of the same kind as rule 1's, three to four times as frequent. Its
update gates stay graded: 73-79% of gate values lie between 0.1 and 0.9,
against 43-53%, and this flips within one training phase when the rule is
swapped. One-step gradient alignment (cosine about 0.6) does not differ.

Questions.
1. Has anyone reported that approximate-gradient rules (sign symmetry,
   feedback alignment, DFA, e-prop, RFLO, KeRNL, target propagation) produce
   less saturated gates, less precise memory, or worse systematic
   generalization in gated recurrent networks than exact gradients do?
2. Which mechanisms make gates or memory units decisive using only
   information local to the unit: bistable cells, gate regularizers,
   saturation-promoting initializations, hard or stochastic gates? For each,
   what was shown on tasks needing memory over 10+ steps?
3. Which studies learn the backward weights of a recurrent network by a
   local rule (Kolen-Pollack, weight mirrors, learned sensitivity weights)
   and compare with fixed random or sign feedback on the same task? Report
   every arm's number, read from the figure if necessary.
4. Are there recurrent units for which exact credit through time is cheap
   and local (element-wise or diagonal recurrence with learned rates) that
   have been shown to compose learned operations on untaught combinations?
5. Negative results for any of the above.

Output. One table with a row per study; then at most five candidate remedies
ranked by evidence, each with the cheapest test in the setting above; then a
list of claims you could not verify from a primary source.
```
