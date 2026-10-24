# Claude response supplied by the user — 24 October 2026

Source: user attachment; retained as design input, not as an execution result.

Yes, the feedback is useful. Three points are clear corrections to my memo or code, one sharpens claims I overstated, and one rightly insists on keeping disclosed limits prominent. I checked the numbers it cites against the saved outputs and they reproduce: learned bank 0.916, frozen 0.467, shuffled 0.563, and system 0.695 balanced held accuracy (the mean over the four worlds).

**1. Parity, not addition, and component vs system: agree.** This is in REPORT.md, but the headline should keep it in front.
- The result is parity transfer for one answer byte, delimited from outside the fly.
- There is no native end-of-answer (EOS) byte, no stopping, and no retention test under later learning.
- The 0.92 belongs to the learned bank; the fly's own system answer is 0.70.

**2. The two-digit extension: agree. This was my error.**
- With only units digits mattering, there are still only 55 units relations and 15 held-out ones.
- `27+35` and `47+85` are two presentations of one relation, so held-out uncertainty stays at the level of relations, not presentations.
- My fresh-stream argument weakens too. A first-exposure error on a seen units relation with new tens digits measures whether the fly ignores the tens digit, not whether it generalises to new relations.
- The memo's claims of "many held items" and "much less chance noise" are wrong.

**3. "Starvation" is a hypothesis: agree.**
- My own native diagnostic contradicts the strong version. The parity bit's residual stayed at 0.41 → 0.22 through pass 20, and the update uses continuous margins.
- The surrogate's 60-pass failure fits "slow learning" just as well as "starvation".
- Collapse: centring the error halved the overlap without improving held accuracy, so collapse isn't shown to limit anything.
- The BP point is also fair. "Even exact gradient fails mod-10" was true only for the settings I tried (H 128, three learning rate and weight decay combinations, up to 3,000 epochs). It doesn't show the data are insufficient.

**4. Replay and arbitration were already tried: agree, and it's an awkward omission, since I built that experiment** (`minifly_cls_dev_20261017`).
- Replay kept addition at 70–85% while multiplication was learned, against 4–6% without replay. Unseen sums stayed at 0–20%.
- The combined answer was sometimes below the better subsystem (53–59% against 56–73%).
- The memo must cite these results and say what would be different now. Candidates: a learned-address bank instead of the recurrent slow core, and arbitration weights learned from errors instead of the earlier rule. Otherwise those steps repeat known results.
- The full historical mechanism registry should be reviewed too, not only the new human catalog.

**5. Post-hoc checker and cost accounting: agree, and I confirmed the bug.**
- In `posthoc_check.py`, a failed runtime-closure check or locked-hash mismatch is recorded but never added to `problems`, so the verdict stays valid. That's fail-open again, in the tool written to be fail-closed.
- It doesn't change this run (both flags pass), but it must be fixed before reuse.
- The 11.2 CPU-hours also leave out the interrupted copy-control runs and the 0.2 CPU-hour determinism re-run.

**Where I'd adjust Codex's recommendation:** the direction is right. Return to ordinary addition with complete native byte output and its own EOS, matched warmup, feedback exposure and CPU, both component and system learning curves, and taught recall and retention. Two cautions:
- **Go straight to native and the comparison may say nothing.**
  - The tens byte of a single-digit sum is the carry, which frozen random features already solve (84–89% held in the screen).
  - The units byte is the hard interaction, and nothing has yet shown it can be learned from about 74 examples.
  - If no learner can do it, learned and frozen both fail. So the first step should be a broader evaluator-side capacity screen: more hidden sizes, learning rates, weight decay and training lengths, plus how many examples are needed. That turns Codex's point 3 into a test.
- **Several things change at once.** Task, output length and own stopping all change together. That's acceptable if the screen and a dress rehearsal confirm each piece, but it's a bigger step than one-change-at-a-time.

Its other recommendation needs no adjustment: measure the bottleneck before adding mechanisms. That diagnostic is cheap. Re-run a dev-world history for LEARN and FROZEN, logging each answer bit's residual, the size of each W1 update, and how fast the native margins grow.

**Proposed actions, in order:**
1. Fix the post-hoc checker so closure and hash failures invalidate the verdict, and add full campaign CPU accounting.
2. Revise the memo: correct point 2, downgrade starvation and collapse to hypotheses, add the CLS v0 replay and arbitration results with what would change, put the parity-not-addition limit in the headline, and adopt Codex's addition-first direction.
3. Run the residual and update-size diagnostic on a dev world.
4. Run the addition capacity screen: complete output with EOS, broad BP sweep, data scaling.
5. Only then design V2: learned vs frozen addressing on ordinary addition, with both curves, recall and retention.

Steps 1 and 2 take under an hour; shall I go ahead with them?