# E20-A stage 2: dry runs and forecast (10 November 2026, written before any stage-2 life)

The predictions are in `E20A_PREDICTIONS.md` (committed at 0ea4ce45, unchanged). The coordinator filed stage 1 as review 9.77 at a87764f5, and on that report gave the go for stage 2. This file adds the stage-2 dry runs and the forecast that the predictions file deferred ("the other three shapes will be dry-run here before stage 2").

## Dry runs (seed 73129, `E11_STAGE_UPDATES=4`, all three at once, 23:15–23:24 UTC, container restarted at 23:14)

| Shape | Exit | Stages | Birth (admitted by) | Against the coordinator's dry receipt in `exp45/archive_e20_dry_v2/` | CPU updates / scoring |
|---|---|---|---|---|---:|
| `E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 ... T 73129` | 0 | 7, completion record | `e11_dev_M_73129.pt`, f21c7fb9… (manifest) | 1 score of 276 differs, at stage 5 (`max97` k = 0 final, 313 here, 314 there); all others identical | 55 / 379 |
| `E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 ... M 73129` | 0 | 7, completion record | `e11_dev_M_73129.pt`, f21c7fb9… (manifest) | every score identical at every stage | 39 / 320 |
| `E16_COMPGAP=6 E16_ARCH=bank2 E16_BOOKDECAY=off ... D 73129` | 0 | 7, completion record | `e17_dev_bank2_nbd_73129.pt`, 8ee5e4e6… (manifest) | every score identical at every stage | 69 / 477 |

- Losses agree with the coordinator's to about the seventh significant figure. The weight hashes differ, as they did for the stage-1 shape: the runs are not bit-identical across machines.
- The only settings field that differs is `birth_admission`, which the coordinator's dry runs predate.
- At this dose every plus-minus memory score is zero or near it, as expected.

The files are archived in `exp45/archive_e20_dry/cloud_20261110/`.

## Launch

Nine lives as a queue, four at a time on four cores, in the order T (73129, 93683, 116089), M (the same), two-bank D (the same). Each command is exactly the brief's. Logs are written to `exp45/logs/e20a_<arm>_cm6_<seed>.log` after the runs end, and each process's environment is read from `/proc/<pid>/environ` while it runs.

## Forecast

Basis:
- **Update CPU:** stage 1's measured 4,950 CPU-s per window-reference life, scaled by the ratio of update times. For T this comes from the qualifier (171 against 144 ms). For M and two-bank D it comes from the dry runs' update CPU (39 and 69 against 41); that dry-run ratio is rough.
- **Scoring CPU:** about the dry runs' figures. Stage 1 measured 307–326 against its dry run's 350.

| Arm | Updates per life | Scoring per life | Per life |
|---|---:|---:|---:|
| T, sign rule | about 5,900 CPU-s | about 380 | about 6,300 |
| M, backprop on the pipeline | about 4,700 | about 320 | about 5,000 |
| Two-bank D | about 8,300 (least certain) | about 480 | about 8,800 |
| Stage 2 in all, nine lives | | | about 60,000 CPU-s |

**Wall time:** about 4.2–4.8 h, against the brief's about 4 h.
