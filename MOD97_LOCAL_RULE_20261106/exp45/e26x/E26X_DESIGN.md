# E26-X: birth weights crossed with the later update (design, written before any run)

**Date:** 12 November 2026. **Anchor:** branch `claude/adoring-cori-b07aog` at `9e429f9a` (E26 stage 3 complete). The harness sources are those of `360e1b8d`, unchanged.

**Authorization.** The owner asked, on 12 November 2026, "Can you run this?" about the next-step proposal in `MINIFLY_E26_RESULT_AND_NEXT_STEPS_20261112.md`, which recommends crossing birth weights and the later update first. That memo is on the owner's Mac and is not in the repository. This design follows the proposal as stated in the owner's message. If the memo's controls or acceptance conditions differ, the memo's prevail, and the difference is to be recorded.

## Question
E26: the sweep arm (S) met both marks jointly in 0 of 3 seeds, the backprop arm (B) in 2 of 3. The gap appears at the first composition checkpoint (stage 3, F2.1). E26 changed the birth and the later update together, so either could be responsible. The question is which one.

## Cells (birth weights × later update; 3 seeds: 73129, 93683, 116089)
| | update S (sweep, local=True) | update B (backprop, local=False) |
|---|---|---|
| **birth S** | SS = E26 life S (exists) | **SB (new)** |
| **birth B** | **BS (new)** | BB = E26 life B (exists) |

- The new cells are 2 × 3 = 6 lives, run by `exp26x_cross.py`.
- The life is `exp26_candidate.life`, unchanged. Only the birth file and the update flag are named separately.
- Everything else is held fixed: the schedule, the data streams (the data digest must equal the E26 lives' for each seed), the panels, a fresh AdamW (decay 0.25, embedding table 0), the scorer, the host, and the full dose (6 stages of 6,144 updates).
- **Cost:** about 5,000 CPU-s per life, so about 8.3 CPU hours. Run three at a time on one thread each, which is about 3 h of wall time.

## Pilot (before the formal runs): can the birth and the update be named independently?
Run at reduced dose (`E11_STAGE_UPDATES=256`), seed 73129, from the full-dose E26 births:
1. `exp26_candidate.py life S` and `exp26x_cross.py life S S` must give identical receipts: equal tensor hashes, data digests and every score. Likewise `life B` and `life B B`.
2. `life S B` must record the S birth's sha256 and run with local=False. `life B S` must record the B birth's sha256 and run with local=True. Both must have the same data digest as the diagonal cells.
3. The four cells must not all be identical. In particular, SB must differ from SS, and BS from BB.

If any of these fails, no formal run starts, and the failure is reported.

## Endpoints (as E26)
Read at the end of life (stage 6), as whole answers (`exact`):
- `within`, out of 600: new instances of taught pairs. Mark: 85.
- Plus-minus: `+-` and `-+` pooled, out of 1,200. Mark: 40.
- Joint pass: both marks in the same seed.

Also reported:
- all four untaught pairs at every checkpoint;
- the stage-3 values, where the E26 gap appeared;
- F1, F3 and dev89.

## Readings, fixed now (the worker's proposal; the memo's conditions prevail)
Two effects are defined, each seed by seed:
- **Update effect** for birth b: U_b = X(b, update B) − X(b, update S).
- **Birth effect** for update u: Bi_u = X(birth B, u) − X(birth S, u).

An effect is said to be **present** when:
- it is at least +5 points on plus-minus in at least 2 of the 3 seeds, **and**
- its mean over the three seeds is at least +5.

The three outcomes:
- **The update carries the gap.** U_S and U_B are both present. Then the sweep's own problem in the composition-learning stage is investigated first.
- **The birth carries the gap.** Bi_S and Bi_B are both present. Then the first question is what differences in ability the high single-step birth scores are hiding.
- **Interaction.** Any other pattern, for example an effect present under one start or update and absent under the other. This is read as an interaction and reported as such, without being attributed to either part alone.

Joint passes per cell (out of 3) and the `within` differences are reported beside these, under every outcome.

## Not in scope
- Byte-by-byte training and autonomous replay come after attribution, each changing one condition.
- E21b's independent continuation and the target-test validations belong to other jobs. They do not depend on this one and are not run here.
