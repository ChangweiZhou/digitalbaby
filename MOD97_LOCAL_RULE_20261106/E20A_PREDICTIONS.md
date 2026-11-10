# E20-A predictions: does the window learner compose when an operand must be remembered? (10 November 2026, written before any life)

**Brief:** `OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md`, Job A, version 2. This session runs Job A only. The review paragraph is `E20A_REVIEW_SECTION_9_77.md`.

**Harness:** the branch head 51a49457, as the coordinator instructed. No existing file is changed by this job.

**Coordinator's instruction for this session (10 November):**
- start from the branch head;
- run the two admission commands first, and both must exit 0;
- run stage 1 only, the three window-reference lives, and report;
- the other nine lives go ahead only if the in-view control works (P20.3), and only after that report.

## Admission (both exit 0 on this machine, 21:26–21:27 UTC)

- **`python3 e20_assay_check.py`: exit 0, passed on all three seeds.**
  - The responder that answers every prompt correctly scores 600 of 600 twins and 1,200 of 1,200 twin items. It also scores 600 of 600 on the leading control, 300 of 300 on `within_hidden` and 300 of 300 on `taught_hidden`.
  - The responder that answers from the last ten bytes alone scores 0 of 600 twins and 600 of 1,200 twin items (one member of each twin), and still scores 600 of 600 on the leading control.
  - The record it writes is byte-identical to the coordinator's `exp45/E20_ASSAY_CHECK.json`.
- **`E16_COMPGAP=6 python3 exp16_integrated.py qualify`: exit 0.**
  - Its record matches the coordinator's `exp45/QUALIFY_E20_MEMORY_PANELS.json` in every field. That includes the memory panels and their hash, the prompt length, the window, the panel sizes, the shortcut ceilings and the composition scores before and after the smoke updates.
  - The exceptions are the two timing fields: here `ms_per_update_mixed` is D 144, T 171, and `s_per_compose_panel600` is 1.06.

Both records are kept as `exp45/archive_e20_dry/cloud_20261110/E20_ASSAY_CHECK_cloud.json` and `QUALIFY_E16_cm6_cloud.json`. The committed files the commands rewrite were restored unchanged.

## Runs, seeds 73129, 93683, 116089

```
# stage 1: the assay's own qualification on a reference (three lives) -- this session runs this now
E16_COMPGAP=6 python3 exp16_integrated.py D <seed>                                         # window reference: backprop, joint stream
# stage 2, only if P20.3 holds and the coordinator says go (nine lives)
E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>            # window learner: sign rule, raw reservoir
E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M <seed>            # backprop on the learner's pipeline
E16_COMPGAP=6 E16_ARCH=bank2 E16_BOOKDECAY=off python3 exp16_integrated.py D <seed>        # windowless two-bank reference
```

If P20.3 misses, the job stops after stage 1 and reports: the reference cannot do the in-view control at these lengths, so the assay cannot speak about memory, and that has to be diagnosed first.

## Predictions (from the brief, copied verbatim; not to be changed)

End of life; plus-minus twins means the `+-` and `-+` twins pooled; item accuracy is whole answers over single items.

- **P20.1 (existence)** The window reference gets both members right on at least 20% of plus-minus twins in at least two seeds.
- **P20.2 (what hiding costs)** The window reference's item accuracy on plus-minus twins is at least 15 points below its accuracy on the same items with leading letters, in all three seeds.
- **P20.3 (the control works)** The window reference reaches at least 70% item accuracy on the leading control for the plus-minus pairs in at least two seeds.
- **P20.4** The windowless reference gets both members right on at most 5% of plus-minus twins in all three seeds.
- **P20.5** The sign-rule learner's item accuracy on plus-minus twins is at or below 0.60 of the window reference's in at least two seeds.
- **P20.6** Backprop on the learner's pipeline is at or above 0.85 of the window reference's in at least two seeds.

Existence (P20.1) and cost (P20.2) are separate verdicts. The window reference is a joint-stream ceiling with its own decay and supervision; the one-factor comparison for the rule is the learner against backprop on its pipeline, both without the store.

**Reading, written in the brief before the result:**
- P20.1 holding with P20.3 says a learner on file can compose with an operand in memory, at the cost P20.2 measures, and the windowless failure belongs to the two-bank design.
- P20.1 missing with P20.3 holding says the window learner composes only in view.
- P20.4 missing would mean the two-bank network can use a hidden operand after all, and version 1's reading of it was wrong.

## How I will score them (fixed now)

Everything comes from the receipt's last stage (checkpoint 6), `scores.memory`.

| Quantity | Definition |
|---|---|
| Plus-minus twins both correct | (`twins['+-'].both_correct` + `twins['-+'].both_correct`) / (`twins['+-'].pairs` + `twins['-+'].pairs`) |
| Plus-minus twin item accuracy | (`twins['+-'].items_correct` + `twins['-+'].items_correct`) / (2 × pairs) |
| Leading-control item accuracy | the same, from `lead` |

- Numbers are given in absolute counts beside the percentages, with the panel each comes from named.
- Whole worked answers are reported, as the memory panels score them.
- On the registered composition panels, the whole answer (`exact`) and the final answer (`final`) are reported side by side.

**Each prediction:**
- **P20.1:** both-correct ≥ 20% in at least two seeds.
- **P20.2:** lead item accuracy minus twin item accuracy ≥ 15 points, in all three seeds.
- **P20.3:** lead item accuracy ≥ 70% in at least two seeds.
- **P20.4:** both-correct ≤ 5% for `D_bank2_nbd_cm6`, in all three seeds.
- **P20.5:** the ratio of twin item accuracy, sign-rule learner over window reference, ≤ 0.60 in at least two seeds.
- **P20.6:** the ratio of twin item accuracy, backprop on the learner's pipeline over window reference, ≥ 0.85 in at least two seeds.

## Dry run (stage-1 shape)

`E16_COMPGAP=6 E11_STAGE_UPDATES=4 python3 exp16_integrated.py D 73129` was run detached at 21:28 UTC and exited 0 at 21:35.

- **Stages:** seven, with a completion record present.
- **Birth admission:** `e11_dev_M_73129.pt` (sha256 f21c7fb9…) was admitted by the manifest.
- **Panel hashes:** the memory-panel hash adb93614… matches the qualify record, and the registered-panel hash cd62110d… matches the coordinator's dry receipt.
- **Scores:** identical to the coordinator's dry receipt (`exp45/archive_e20_dry_v2/e16_life_D_cm6_dry4_73129.json`) in all seven stages.
- **Weights:** the hash differs (1ab48cb3… here, afe6e8fc… there), so the weights are not bit-identical even though every score is.
- **Memory scores:** at this dose every plus-minus memory score is zero, as expected.
- **CPU:** 41 CPU-s of updates and 350 of scoring.

The files are in `exp45/archive_e20_dry/cloud_20261110/`. Only the stage-1 shape was dry-run now; the other three shapes will be dry-run here before stage 2, if stage 2 goes ahead.

## Run identity

| Item | Value |
|---|---|
| Machine | `Linux vm 6.18.44-fc-v338 #1 SMP @0 x86_64 x86_64 x86_64 GNU/Linux`; 4 cores; Intel Xeon at 2.10 GHz |
| Container | freshly restarted at 21:25 UTC |
| Software | Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6; one thread per process |
| Harness | 51a49457 |

Each process's environment will be read from `/proc/<pid>/environ` while it runs.

## Forecast, stage 1

| Item | Forecast | Basis |
|---|---:|---|
| Updates, per life | about 5,300 CPU-s | 36,864 updates at the qualifier's 144 ms per mixed update for arm D |
| Scoring, per life | about 350–500 CPU-s | 50 CPU-s per checkpoint in the dry run, idle, seven checkpoints; more under load |
| Per life | about 5,700 CPU-s | |
| Stage 1 in all | about 17,000 CPU-s | |
| Wall time | about 1.6–1.7 h | three lives on four cores |

The brief's estimate is about 1.5 h.

## If the container restarts

A restart kills every process. A life killed before its completion record has not finished. It will be relaunched from its birth with the same command and seed, its partial receipt kept as `*_killed.json`, and the report will say so. No finished life is rerun.

## Push target

`codex/minifly-consolidated-20261023`: pull with `--rebase` first, never force, and push at most once an hour while runs are in flight.
