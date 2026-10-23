# ERROR versus R_center: infrastructure blocker report

Status as of 2026-10-05 00:33:02 UTC: **the fixed 64-world official run is blocked under its unchanged host-budget admission rule. No terminal efficacy result is available.**

## Recoverable progress

- All three independent design–audit–test cycles passed before the official run.
- Worlds 320001–320026 completed and have verified private completion backups: **26 of 64**.
- World 320027 was observed to finish natively before the execution workspace became unavailable. Its output was not preserved in the latest durable checkpoint. That checkpoint contains its prebirth reservation, with the complete **900-second** attempt charge retained. It is classified as an infrastructure interruption independently of outcomes and is not counted as a completed world.
- Worlds 320028–320064 had not started. World 320027 has not been rerun.

Recovery verified all 5,170 files in the latest checkpoint payload, all 166 frozen source files, all 24 bound acceptance dependencies, and all 3,406 receipt parts from the 26 completed worlds. No hash mismatch was found. The archived evidence and scientific settings remain unchanged.

## Why the run cannot resume under the current contract

The frozen host-recovery rule treats a session without a preserved normal closure conservatively. Its charge is the larger of the last recorded effective charge and the prior cumulative charge plus elapsed time from that session's start to recovery. This is conservative accounting, not a claim that computation ran throughout the interruption.

For the recovery snapshot above:

| Quantity | Seconds | Hours |
| --- | ---: | ---: |
| Conservative recovered host charge | 27,596.21 | 7.6656 |
| Projected total host charge for the unchanged remaining plan | 67,155.95 | 18.6544 |
| Registered host cap | 64,800.00 | 18.0000 |
| Amount by which the projection exceeds the cap | 2,355.95 | 0.6544 |
| Projected total worker charge | 28,076.87 | 7.7991 |
| Registered worker cap | 43,200.00 | 12.0000 |

The host projection retains the original packing, transfer/readback, analysis, retry and other reserves. It already fails before adding any extra restart costs. The worker cap alone would pass; that does not override the host gate. Subsequent waiting cannot make this particular conservative projection smaller.

The preserved current-session journal ends at session start and has no normal closure event. The latest checkpoint still marks that session active, and the interrupted runner reported that closure could not be verified. A changed machine boot identity establishes a different execution environment, but the frozen rule provides no exception that uses it to shorten the unknown interval. No substitute accounting rule was applied.

## Scientific interpretation and disposition

The planned unit of inference is all 64 fixed worlds. The registered terminal analysis requires that complete cohort, so no partial-cohort efficacy analysis or terminal hypothesis classification has been produced. These 26 completed worlds do not establish ERROR's success, failure, noninferiority, or reduction in harm relative to R_center.

The run remains blocked. No cap extension, replacement world, rule adjustment, rescaling, extra warm-up, or adaptive sample extension has been made. Completed worlds will not be rerun. The unpreserved attempt remains charged rather than being erased or treated as successful.

This report is published first to document the blocker. The initial code, protocol and qualification release is available at [commit 79c13ab8](https://github.com/ChangweiZhou/digitalbaby/tree/79c13ab8cede0f9fd7bda6bd45cc176735247702/studies/error-completion-20261004). Its public projections are inspectable scientific artifacts, as described there; private recovery checkpoints and administrative records are excluded from publication.

SPDX-License-Identifier: GPL-3.0-or-later. Inherited attribution remains as documented in the study's license provenance.
