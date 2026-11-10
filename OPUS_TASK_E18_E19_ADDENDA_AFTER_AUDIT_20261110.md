# Addenda to the E18 and E19 jobs after the 10 November source audit (10 November 2026, 15:00 EST)

For the workers running E18 and E19. Nothing here changes a prediction, a seed, or a run in flight; finish what is running as registered. These are additions to the report, and for E19 one control arm. Source: `PROJECT_DOCUMENTS/MINIFLY_E16_E19_HOSTILE_SOURCE_AUDIT_20261110.md`, response in `PROJECT_DOCUMENTS/AUDIT_RESPONSE_20261110.md`.

## E18 (births with a next-byte target at every bank)

1. **B3 is a whole-package arm.** Measured by the coordinating session at initialisation: under B3 the embedding rows of operators, task codes and distractor letters receive no gradient at all (11 of 45 symbol rows are reached); under B0, B1 and B2 all are. Say so in the report beside P18.6, and do not read a B3 failure as a statement about the bank.
2. **Report embedding movement.** For each of the twelve finished births, load the saved weights and the initial network of the same seed and settings (`E.new_net(seed)` under the birth's environment) and report, by symbol role (digit, `=`, operator, task code, distractor letter), the mean distance of the rows from their initial values relative to the mean initial row norm. `exp16_integrated.book_shift(m, book0)` at the current branch head does this. This is read-only.
3. **Say what differs between arms.** B0 to B3 differ in more than the relay: the auxiliary head and its parameters, the feedback form, and decay (0.30 for the backprop birth B1, 0.25 for the rule births). List these beside the verdicts. The one-factor controls (`E16_HEADS=aux`, `E16_FIXEDCODES=on`) exist now and are a later job; do not run them.
4. Births finished before this addendum have no credit census or head-1 loss in their receipts; that is expected. Do not rerun them.

## E19 (forward-only two-bank learner)

1. **One control arm: Stage A at the rule's decay.** P19.3 compares rule births at decay 0.25 with backprop births at 0.30, so its ratio does not isolate weight signs. Add three backprop births at decay 0.25, same seeds, files `e19_dev_kbank2_bptt25_<seed>`, and report R1 against both. The registered P19.3 is judged as written and marked confounded; the decay-matched ratio is reported beside it.
2. **Assert the rates.** Before any trace is computed, assert that the network's buffers `R1` and `R2` equal `arch19.rate_vector(C)` and that the group index used by the traces matches them. Add this in a new commit; say that births already finished used the default rates, which the assertion would have passed.
3. **Streaming inference.** The gradient tests show the rule is forward-only. Scoring rebuilds the state from the whole prefix for every output byte. Add a check that an incremental scorer (state carried from byte to byte, reset between items) gives the same answers as `E.score` on one panel, and report it as its own row.
4. **Three rows, not one.** Report separately: the rule equals backprop through time (the tests); the learner learns (the births, in absolute numbers as well as ratios, and at k = 20 as well as 12); the rule is cheaper (measured). The control arm R2 allocates the deep-trace array without using it, so its memory does not show the saving from removing the traces; say so or measure it properly.
5. Nothing in E19 isolates the removal of the gated write: the change from the E17 network replaces several parts at once. Say so beside P19.2.
