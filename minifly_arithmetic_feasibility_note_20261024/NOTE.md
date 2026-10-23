# Is basic arithmetic from examples too much to hope for? (note, 2026-10-24)

Discussion note. Nothing built or run. Evidence base: Codex results through V6-R1, the v6 memo, and `human_learning_mechanisms_20261023/CATALOG.md`.

## Answer
Not too much to hope for, but split into three goals with very different odds.

1. **Memorising taught single-digit facts: likely.** The memory has stored random 4-byte values at 35-100% bit accuracy, and single-digit answers are short, so stopping matters less. Remaining work is engineering: lower cross-talk, balanced end marker, equal write strength for old and new facts.
2. **Generalising to unseen single-digit pairs: plausible (better than even, my judgement).** Our earlier GRU diagnosis: a magnitude-structured (tally) input generalised 98-100% while digit symbols were memorised. Engineered-prior versions (STABLE, COV-R) confirmed addition on 0-4 once digits, roles and ordinal outputs were supplied. The open piece is learning that ordering from data. Human literature supports it (number-line and number-neuron findings, grade B).
3. **Multi-digit addition with carries on unseen problems: the real open question.** Unconstrained small networks with backprop learn it routinely; under our constraints (local credit, no attention, lean) I found no precedent. Real chance, no guarantee. Our zero results say little about feasibility because every full run failed an earlier prerequisite (storage cross-talk, then a recurrent state that carried noise instead of input).

## Next steps (in order)
1. **Input-vs-noise gate on every recurrent candidate:** same input with different noise must give states closer than different inputs with the same noise, checked before any learning. V6-R1 failed this by about 70x undetected. Codex's rescale is its first test.
2. **State on a toy task with the answer table removed from the question:** running parity / sum mod 10 feeding an ideal evaluator-only lookup. Fail means change the credit approach (latent-cause / clone-structured route); pass means integrate. If it fails after a few principled variants, stop instead of tuning.
3. **Learn the magnitude code from data** (topographic neighbourhood learning over byte-value units); test on held-out single-digit pairs (goal 2).
4. **Training schedule:** apply the strongest human findings (spacing, interleaving, retrieval practice with feedback, worked examples, self-explanation; grade A or near in the catalog). Our runs used 128 facts x 32 massed passes. The fast store's ~1 h decay makes spacing mechanistically relevant; for a different learner this is a hypothesis.
5. **Integrate:** procedure plus memory (counting-to-retrieval shift), then 2-digit, 3-4-digit and longer, then subtraction and decimals.

## Decisions that most affect the odds
* Local auxiliary predictor for representation credit, all answers still native.
* Keep exploration noise out of the memory address.
* Worked-example text in the data.
* Faster harness (full histories cost ~385 CPU-s each; toys are cheap).

Pass and kill criteria for steps 1-4 to be fixed in advance once these are approved.
