# Task for Codex: E16-W, does the window net also forget compositions when a later family arrives?

**Date:** 8 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (your `exp8_life.py` with the E13/E14/E14-S knobs; standing births `exp45/e8_dev_*` for seeds 73129, 93683, 116089). **Purpose:** E16 (`E45_REPORT.md`, section E16) put the cell learner on one stream of three families (mod-97 operations with distractors, then the taught operator pairs, then mod-73 operations) and found that each family is acquired within 1–9 points of backprop but earlier families are lost when the stream changes form: the compositions fall from 84–87% to 46–57% during the two mod-73 blocks, with the earlier family in the replay half of every batch. E16-D (running here) tests whether the replay sampling or the local rule carries that. This job is the mirror from your side: E8 has only ever learned compositions *last*. Teach it a later family after the composition stage and measure whether the compositions survive, under E8's own replay and under a balanced variant. No new mechanisms, no new tasks, no new panels; one schedule option and one replay option.

## The schedule option

Add `E8_SCHEDULE=late_family` (default `blocked` unchanged): the four mod-97 operations in sequence as now (4 × 6,144 updates), then the composition stage as now (8,192 updates on the twelve taught pairs), then the four mod-73 operations in sequence (4 × 6,144), each a stage. Score every panel after every stage as now: all eight operations, the four held pairs, within. The composition stage's presented items must enter the replay buffer (1,024 of them, as any other stage) so that compositions can be replayed during the mod-73 stages; today `name != "compose"` excludes them from both store and buffer, and that exclusion is what this schedule removes. Record exactly what enters the buffer.

## The replay option

Add `E8_REPLAY_SAMPLING=balanced` beside the existing `uniform` (default): keep the buffer per stage (per family), and draw each replayed item from a stage chosen uniformly among those in the buffer, then uniformly within it. The buffer sizes are unchanged (1,024 per stage). Default path bit-identical; one parity check with hashes.

## Arms, three standing seeds

- **C_uni**: sign rule, decay 0.25, share 1/2, batch 64, `late_family`, uniform replay (E8's replay, E13-A1's settings).
- **C_bal**: the same with balanced replay.
- **D**: backprop ceiling at its standard setting on the `late_family` schedule (joint ground truth of everything seen so far, as arm D already does).

Nine lives plus the parity checks. Report, after the composition stage and after each mod-73 stage: the eight operations, each held pair, within; one table of held-pair mean against stage for the three arms; the mod-97 operations' retention through the mod-73 stages.

## Predictions to test (write them in `E16W_PREDICTIONS.md` before any run)

1. Under uniform replay the held-pair mean falls by at least 15 points from the end of the composition stage to the end of the fourth mod-73 stage in every seed (E16's cell learner lost 30–41 in two blocks). If it falls by less than 10, the composition forgetting in E16 is a property of the cell learner, not of this replay.
2. Balanced replay at least halves that loss in every seed.
3. The mod-97 operations stay at or above 90% through the mod-73 stages under both replays (E8 retained eight operations at 94% with the same buffer), i.e. the forgetting, where it occurs, is of the composition form, not of operations.
4. Arm D holds everything: held pairs within 3 points of their post-composition value after the fourth mod-73 stage.

## Rules

- Predictions first, with the exact commands, a forecast from your E14-S measured pace under the actual load (nine E16-D processes are running on this machine until about 13:05 EST), and the parity hashes.
- Keep the shortcut ladder, panels and checkpoints. No tuning. If a part fails, report and stop it.
- Deliver `E16W_REPORT.md`, receipts under `exp45/`, the edited script, and one paragraph in `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md` as section 9.31 (9.30 is reserved for E16-D). Report every seed separately.

## Budget

Nine lives of 13 stages each at hidden 512 (about 1.5× an E8 life), roughly 5 CPU-hours; under two hours of wall time with six workers once the E16-D processes finish. The qualified inference acceleration may be used; record that it was.

## Amendment, 8 November 2026 12:15 EST (after Codex's review of this brief; before the nine lives start)

Codex's objections are correct and the arms change accordingly. (i) With 1,024 entries per stage, "choose a stage uniformly, then an item" has the same expected share per stage as "draw uniformly from the whole buffer"; the balanced arm as written changed only repetition and batch correlation, not exposure, so prediction 2 had no intervention behind it. (ii) The backprop arm draws from every item seen so far, so in the mod-73 stages compositions are about 62% of its batches against about 6–10% of the local arm's; it is a ceiling, not a comparison of rules. (iii) A window net that keeps compositions would narrow the cause of E16's loss to the cell learner's side (window length, distractors, order, dose, the store) but would not prove the cell is the cause; prediction 1's second sentence is withdrawn and replaced by "localizes, does not prove".

**Arms, amended (three standing seeds):**
- **C_uni**: unchanged (sign rule, decay 0.25, share 1/2, batch 64, `late_family`, E8's uniform replay). Answers whether the window net forgets compositions across forms at all.
- **C_share**: the replay half of every batch during the mod-73 stages is drawn half from the composition stage's buffer and half from the operation stages' buffers (so compositions are 25% of every batch, up from about 6–10%); everything else as C_uni. Log the realized exposure (items per family per stage) in the receipt. This is the real share intervention; `E8_REPLAY_SAMPLING=balanced` as first specified is withdrawn and need not be implemented.
- **D_ceiling**: backprop with joint ground truth as before, labelled a ceiling, its composition share recorded.
- **D_matched**: backprop on exactly C_uni's batches (same replay buffer, same draws: run the C_uni data pipeline with the backprop update), so that the rule is the only difference from C_uni. If D_matched keeps the compositions and C_uni does not, the rule is implicated; if both lose them, the replay is.

Twelve lives. Predictions re-locked in `E16W_PREDICTIONS.md` before launch: (1) C_uni loses at least 15 points of held-pair mean over the four mod-73 stages in every seed; if under 10, E16's composition loss is localized to the cell learner's side and not to E8-style replay. (2) C_share loses at least a third less than C_uni in every seed; if it loses as much, exposure share is not the cause. (3) D_matched keeps the held pairs within 10 points of their post-composition value; if it loses as much as C_uni, the replay data, not the rule, is the limit. (4) mod-97 operations stay at or above 90% under both local arms. Budget about 6.5 CPU-hours. The default-parity reproduction already running is kept and counts.
