# Task for Codex: E16-WR2, the window net with observed replay targets and one label-free total-budget reservoir; E8's demonstration rerun the same way

**Date:** 8 November 2026, 18:30 EST. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (your `exp8_life.py` with the E16-WR knobs). **Scheduling:** E16-R3 (three `exp16_integrated.py` lives) holds the machine until about 19:35 EST; prepare now, launch when `ps` shows none. **Purpose:** E16-WR showed that self-generated replay targets run away with coverage (81–91% wrong at 16,384 slots) and that observed targets break the loop but still lose 14–20 points of composition at per-stage capacity 16,384. The cell learner closed the corresponding gap with one label-free reservoir of 65,536 entries over the whole life (E16-R2). Do the same on the window net, and rerun E8's original demonstration with observed targets so the record's headline persistence result no longer rests on pseudo-rehearsal.

## Part A: late-family schedule, observed targets, one total-budget reservoir

Add `E8_REPLAY_SAMPLING=total` with `E8_RESERVOIR_CAP=C`: a single reservoir over every presentation of the life (one counter, Algorithm R, no stage or family key in the replay path; items carry their observed answers), replay half drawn uniformly from it. Arms, three standing seeds, `late_family` schedule, sign rule, decay 0.25, batch 64, `E8_REPLAY_TARGETS=observed`: C = **16,384** and **65,536**. Six lives. Comparators (not rerun): E16-WR observed 16,384 per stage, `D_ceiling`. Record distinct counts, the reservoir's composition by stage (for reading only), realized draw counts.

Predictions: (1) at 65,536 the held-pair loss from the post-composition value to the end is at most 8 points in every seed (per-stage observed 16,384: 14–20), and the mod-97 operations stay at or above 93%; (2) the total-budget 16,384 arm is within 5 points of the per-stage observed 16,384 arm on both endpoints (the label does nothing, as on the cell learner); (3) acquisition within 3 points of E16-W's `C_uni` post-composition value.

## Part B: E8's demonstration with observed targets

E8's original schedule (eight operations in sequence, then composition), arm C at the adopted settings (decay 0.25, share 1/2, batch 64, buffer 1,024 per stage as in E13-A1) with `E8_REPLAY_TARGETS=observed` and otherwise unchanged, three standing seeds. Three lives. Comparator: E13-A1 (self targets, same settings).

Predictions: (4) retention of the eight operations after composition within 2 points of E13-A1 (the operations' self targets were about 95% right, so little changes: 94–95%); (5) held composition after the composition stage at or above E13-A1's in every seed (81 / 82 / 85) by at most 5 points (compositions are learned last here, so the self-target loop had little time to act); the point of Part B is to establish the observed-target baseline for the demonstration, not to expect a large gain.

## Rules

Predictions in `E16WR2_PREDICTIONS.md` first, with the forecast from your E16-WR pace; default parity for the new knob with hashes; no tuning; every seed; `E16WR2_REPORT.md`, receipts under `exp45/`, review paragraph as `E16WR2_REVIEW_SECTION_9_40.md` (do not edit the review). Budget nine lives, about 4 CPU-hours, under an hour of wall time with nine workers on the free machine.
