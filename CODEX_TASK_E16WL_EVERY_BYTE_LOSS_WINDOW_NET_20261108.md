# Task for Codex: E16-WL, the every-byte loss on the window net, under both rules

**Date:** 8 November 2026, 22:10 EST. **Branch:** `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Scheduling, new rule:** before launching any life, create `/tmp/MINIFLY_MACHINE_LOCK` (an absolute path, shared by every checkout on this machine; corrected 22:30 EST from a repository-relative path, which would have been a different file in each checkout) containing your job name and the time, only if no such file exists; remove it when your last life exits. I do the same (E16-A and the every-byte-loss twin hold it until about 23:30 EST tonight). No producer launches while another's lock exists; this replaces the idle-time check, which failed tonight when E16-A and E16-WR2 started within a minute of each other.

**Purpose.** E16-L (`E45_REPORT.md`, section E16-S and E16-L) removed the answer mask on the cell learner: next-byte loss on every byte of prompt and answer. Under the sign rule the learner then fails to acquire addition and subtraction at all (3–5% at k = 12 after their blocks), while the compositions are acquired and lost. Its backprop twin under the same loss is running here tonight. Generalize across learners: does the window net fail the same way, under the sign rule and under backprop, on E8's schedule?

## Runs

Add `E8_LOSS=answer|all` to `exp8_life.py` (default `answer`, bit-identical; `all` masks every real byte of prompt and answer, padding excluded; the store's write mask follows the loss mask). Arms, three standing seeds, E8's original schedule (eight operations then composition), observed replay targets (`E8_REPLAY_TARGETS=observed`, the E16-WR2 Part B setting) so that the loss is the only change from Part B: **C_all** (sign rule, decay 0.25) and **D_all** (backprop at its standard setting). Six lives. Comparators: E16-WR2 Part B (sign rule, answer loss) and E8's arm D.

## Predictions (lock in `E16WL_PREDICTIONS.md` before launch)

1. Under the sign rule with the every-byte loss, the mean of the eight operations after the eighth stage is below 60% in every seed (answer loss: 94–95), and add97 and sub97 are below 20% (the cell learner's add and sub were 3–5% at k = 12; the window net has no distractors, so the floor may be higher).
2. Under backprop with the every-byte loss, the eight operations reach at least 85% in every seed: the objective dilutes the answer signal but backprop's credit reaches it. If backprop also fails (below 60%), the objective, not the rule, is the limit on both learners.
3. Held composition after the composition stage is lower under the every-byte loss than under the answer loss by at least 15 points for the sign rule; for backprop by at most 10.
4. Per-byte loss curves: report the mean loss on prompt bytes and on answer bytes separately per stage (log both), so that "swamped" is measured rather than inferred.

Rules as before: predictions first with the forecast from your pace; default parity with hashes; no tuning; every seed; `E16WL_REPORT.md`, receipts under `exp45/`, review paragraph as `E16WL_REVIEW_SECTION_9_50.md`. Budget six lives, about 3 CPU-hours, under 40 minutes of wall time on a free machine.
