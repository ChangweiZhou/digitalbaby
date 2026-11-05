# Memo: round 31 — first clone-structured (CSCG-style) state learner: negative, informative (2026-11-05)

Evidence: dev. People-only MiniWorld: 3 people, 4 places, 300 questions per world, 3 seeds. Prequential: retrain every 25 questions on the prefix, score the next answers before they are seen. Code: `minifly_review_experiments/round31_cscg_20261105/`.

## A. Monolithic cloned HMM (`chmm.py`)

One latent chain over the token stream, C clones per token, Baum–Welch counting (no backprop). Answer probability by forward filtering.

| clones C | late top-1 (q ≥ 150), seeds 1/2/3 | late answer bits |
|---|---|---|
| 2 | 0.35 / 0.27 / 0.24 | 1.99–2.03 |
| 8 | 0.35 / 0.25 / 0.26 | 2.00–2.06 |
| 32 | 0.35 / 0.23 / 0.21 | 2.06–2.10 |

**References:**

| reference | score |
|---|---|
| chance | 2.0 bits, top-1 0.25 |
| last-mentioned place (shortcut) | top-1 0.48–0.51 |
| round 23/24 count learner | 0.89 |

**Reading: chance level, and more clones do not help.** A single latent chain must represent the *joint* state of all people (4³ combinations × sentence position). That combinatorial explosion is the binding problem in HMM form. Scaling clones alone is not a scalable answer here.

## B. Factorized, entity-centric cloned HMM with a shared schema (`fchmm.py`)

- Every content token gets its own chain over the lines that mention it, with itself recoded as SELF.
- One transition model is shared by all chains. This generic "self-indexing" trick does not say which tokens are people.

**Result:** late top-1 0.30 (C = 4, 8 EM iterations; also C = 8, 30 iterations), i.e. about chance.

The diagnostic shows learning is happening, but slowly:
- likelihood per token falls 0.476 → 0.409 over 30 iterations;
- on one probe, the correct place is ranked first at 0.33, against 0.29 / 0.19 / 0.15 for the others.

## Conclusions

1. **Off-the-shelf latent-state learning does not solve multi-entity tracking.**
   - The monolithic version explodes combinatorially.
   - The factorized version is the right shape, but plain Baum–Welch from random initialisation learns the "carry the last place to the question" memory far too slowly at this data size. CSCG papers rely on many EM passes plus Viterbi refinement, and much longer sequences.
2. **The factorization (one chain per entity, shared schema) is the generic form of what our hand-built PLACE store did.** It is worth pursuing, but it needs a better learning procedure.
3. **Transfer across worlds requires emission remapping**, since place symbols are new each world. Raju et al. 2024 reuse the transition graph and relearn emissions. This is not implemented yet.

## Next

1. Factorized CHMM with **Viterbi-EM refinement and more passes**, plus a **longer-life** check: does it reach the 0.5 shortcut, then beyond?
2. Add **emission remapping for new symbols** (schema reuse) and run the fresh-vs-experienced transfer test.
3. In parallel, keep the round 20 GLN + recall baseline as the bar on answer bits.
