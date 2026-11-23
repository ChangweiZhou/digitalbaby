# NET13 report: consolidated learner, and rule cells gating every readout (23 November 2026)

Branch `claude/newnet-net13-sei232` from `claude/magical-albattani-sei232` @ 32db8c13. This is development level work on one machine,
with **seed 1 only and cue=1 only**. No gate, seal or outside review is claimed. The plan and criteria were written and committed before
the runs (`NET13_PLAN.md`).
New files: `net13.py`, `check_net13.py`, `net13_runs/` (run_full.sh, full_s1_rcall0.log, full_s1_rcall1.log, full_res.jsonl).
No existing file was changed. Learning is local throughout (no backprop). Each life took about 26-27 min on one core; the two ran in parallel.

## Task 1: net13 = net10 K8 + lag cells (W=8) + direct readout + gated relay + rule cells (rc=64, rc_gate=1)
- Every other net12 / net12_rc option is gone. The docstring describes the cells and the local rules.
- Short check (`check_net13.py 300`): 3,600 bytes, including 400 with learning off. net13 and net12_rc ran in separate processes,
  because net7's C library has a process-wide random state: two nets in one process disturb each other, and a first in-process check
  showed spurious differences.
  - Largest difference of the summed outputs: **0**.
  - Answers that differ: **0**.
- Full size (t23_life_rules.py net=13 struct=K8 cue=1 seed=1) **reproduces net12_rc exactly**. Every probe line and phase line of
  `full_s1_rcall0.log` is identical to `rc_runs/full_A1.log` (diff, ignoring the elapsed seconds). Result:
  - succ at the end 0.822;
  - bits 777,024;
  - text 0.348 -> 0.240.

## Task 2: rc_all=1 (the rule cells also set a gain for each of the 33 other compartments)
- Compartment c is summed with gain g_c = 1 + Gc[c] . s.
- The gain learns by Gc[c] += eta_r / ncomp * (e . O_c) * s, the relay gate's rule.
- The gains start at 1, and no new parameter is added.

| run (t23 cue=1 seed 1) | bits total (input / text / answer) | text after text (17k) -> after succ (23k) -> end (29k) | succ after succ -> after copy (26k) -> end | copy / rev end | first-met P0 copy b / c, P1 rev b, P3 succ b |
|---|---|---|---|---|---|
| net12_rc A1 (reference) | 777,024 (564,912 / 143,714 / 68,398) | 0.348 -> 0.222 -> 0.240 | 1.00 -> 0.98 -> 0.82 | 1.00 / 1.00 | 0.995 / 0.825, 0.948, 0.715 |
| net13 rc_all=0 | 777,024 (564,912 / 143,714 / 68,398) | 0.348 -> 0.222 -> 0.240 | 1.00 -> 0.98 -> 0.82 | 1.00 / 1.00 | 0.995 / 0.825, 0.948, 0.715 |
| **net13 rc_all=1** | **762,455** (559,174 / 143,576 / 59,705) | 0.352 -> 0.250 -> **0.238** | 1.00 -> 0.99 -> **1.00** | 1.00 / 1.00 | 0.995 / 0.825, 0.948, **0.776** |

### Text probe during and after the succ phase (lines 17k-29k), rc_all=0 / rc_all=1

| 17k | 18k | 19k | 20k | 21k | 22k | 23k | 26k | 29k |
|---|---|---|---|---|---|---|---|---|
| .348/.352 | .293/.296 | .252/.267 | .240/.262 | .233/.249 | .228/.241 | .222/.250 | .249/.251 | .240/.238 |

- The text is lost in the first 3,000 succ lines in both runs.
- rc_all=1 slows that loss by about 0.01-0.03 at each probe up to 23k.
- By the end of the life the two runs are the same (0.240 / 0.238).

### Against the prediction
| criterion | needed | got | met? |
|---|---|---|---|
| text at the end | >= 0.30 | 0.238 | no |
| falsifier | text at the end <= 0.26 | 0.238 | **falsified** |
| succ at the end | >= 0.81 | 1.00 | yes, better |
| copy / rev at the end | >= 0.99 | 1.00 / 1.00 | yes |
| total bits | <= 777,024 | 762,455 (-1.9%) | yes, better |

**Seed 2 was not run.** The plan made seed 2 depend on a pass at seed 1, and seed 1 failed.

## Verdict
1. **Consolidation done.** net13 (rc_all=0) is bit-identical to net12_rc lagro=1 relay=1 rc=64:
   - on the 3,600-byte check;
   - over the whole full-size t23 life (777,024 bits; succ at the end 0.82; text 0.35 -> 0.24).
2. **rc_all=1 is refuted for its stated purpose.** Text at the end is 0.238, at or below the 0.26 falsifier. It does not keep the text:
   the text is overwritten inside the compartments during the succ phase, and a per-compartment gain cannot undo that.
3. **Unpredicted side effect (seed 1 only, not confirmed):** with rc_all=1 the learned succ mapping is fully kept to the end:
   - succ at the end 1.00 against 0.82; the fall during the last rev phase is gone (27k: 0.98 against 0.90);
   - first-met succ b 0.78 against 0.72;
   - the life costs 762,455 bits against 777,024 (-1.9%; answer bytes 59.7k against 68.4k).

   This result was not the prediction. It needs seeds 2-3 before it counts, and I did not run them.

## Limits / NOT_RUN
- Seed 1 only. cue=0 not run. No toy run and no battery tests (natural text, keeping) with net13.
- The learned gains were not logged, so I cannot say which compartments turned down where.
