# E21b: qualify resume on the merged harness (11 November 2026)

> **Status.** Engineering job, no experiment and no claim about learning. Written after Codex's audit of 11 November (`PROJECT_DOCUMENTS/MINIFLY_HOSTILE_AUDIT_E20B_STEP0_E18_E21_20261111.md`), which asks for exactly this, so it may start without a further audit. Its result is audited before resume is enabled.

## Why

E21 delivered lives and births that survive a restart, and passed 47 checks on its own branch. That branch started from an older commit. The default branch now carries a three-way merge: the earlier default branch, E21, and the options of the E18b/E18c/E23 branch. The 47 checks have not been run on those sources. Codex also found three gaps in E21 as delivered. Until this job is done the harness refuses to resume.

## Where things stand (read `review 9.98` first)

- **The merged harness** is `MOD97_LOCAL_RULE_20261106/exp16_integrated.py` on the default branch at the commit that carries this brief.
- **Added by the coordinator in the merge:**
  - `E16_RESUME=1` is refused unless `exp45/RESUME_QUALIFIED.json` names exactly the hashes of the five source files (`resume_gate`, `source_hashes`). `E16_RESUME_UNQUALIFIED=allow` lets a run resume without it; that is for this job, and it is written into the receipt's `resumed` entries.
  - The configuration a checkpoint is bound to now includes those source hashes, the torch version and the default dtype.
  - Resume is refused for `E16_FARCTRL=shuffle`, whose private generator is not in the checkpoint.
- **Checked locally by the coordinator** (`exp45/HARNESS_MERGE_CHECK.json`): the existing check scripts pass; a dry two-bank birth and life equal the pre-merge ones bit for bit; a dry window life (sign rule, raw store, drifting stream, online admission) equals the pre-merge harness's; each of the two lives, killed and resumed, equals its uninterrupted run. That is two command shapes at one tiny dose. It is not a substitute for the 47 checks.
- **Codex's three gaps, as the coordinator found them:**
  1. *What a checkpoint is bound to.* As delivered it omitted the source hashes, the torch version and the dtype. Those three are now in. The list of supported command shapes and the gate's use of it are yours (item 4).
  2. *A torn final receipt.* Confirmed by fixture: with a final birth receipt cut short beside valid weights and a valid checkpoint, and resume asked for, the run stops with a JSON parse error before any checkpoint is looked at. It does not accept the torn receipt, and it does not recover either. Yours (item 2).
  3. *The far-head shuffle.* Its private generator is not checkpointed. Resume is refused for it.

## The job

Work on your own branch from the default branch's head. Edit only the resume code of `exp16_integrated.py` and `e21_resume_check.py`. Without `E16_RESUME`, every receipt and every weights hash must stay what the default branch's head gives; your own checks prove it.

1. **Rerun all 47 checks on the merged sources,** in a fresh test directory, with `E16_RESUME_UNQUALIFIED=allow` where a check resumes. The "unchanged when not resuming" checks compare against the default branch's head as it is when you start, for the seven command shapes of the E21 brief.
2. **Close the final-publication gap.** A receipt is a result only if it parses and carries its completion record. A final receipt that is torn (cut short by a kill while it was being written) beside valid weights and a valid checkpoint must be set aside and the run resumed from the checkpoint; it must never be read as a result, and never stop the run with a parse error. The same for a life. Write the final receipt by temporary file and rename, as the checkpoints already are.
3. **Add these fixtures**, each shown able to fail:
   - a source file changed after the checkpoint: refused, naming the file;
   - a checkpoint presented to a life from another birth: refused;
   - a changed dose (`E11_STAGE_UPDATES`, `E11_DEV_UPDATES`): refused, naming the setting;
   - a checkpoint and a `.ckpt.prev` that belong to different runs or to non-consecutive stages: refused;
   - a kill during the final publication of a birth and of a life: resumed from the checkpoint to the same receipt and weights hash as an uninterrupted run;
   - a different torch version recorded in the checkpoint: refused (edit the recorded value in a copy to show it).
4. **A list of what is qualified.** `RESUME_QUALIFIED.json` names the command shapes that passed (arm, architecture, replay mode, stream, admission, loss, heads, birth rule, credit through time). The gate then refuses to resume a run whose shape is not on the list. Shapes E21 did not test (`uniform`, `balanced`, `share`, `reservoir`, `total` and `oracle` replay, `E16_FREEZE`, `E16_HEADS=aux` and `relay`, `E16_RSHARE` other than one half, the far-head options) stay off the list unless you test them.
5. **Write `exp45/RESUME_QUALIFIED.json`** only when everything above passes: the five source hashes, the torch version, the list of shapes, the hash of the check record, the host. One commit on your branch.

## What comes back

`E21B_REPORT.md` (what was run, what failed first and why, what is on the qualified list and what is not), the check record, the qualification file, the harness diff. If any of the 47 fails on the merged sources, stop and report it before changing anything to make it pass.

## Limits

- At most three processes on a four-core container. E21's own acceptance run took over an hour; forecast from its record and replace the forecast with your measured pace.
- Containers restart. Make the check runner keep finished jobs, as E21's rerun did.
- Nothing partial goes to the default branch. The coordinator reruns the local smoke test on your branch and merges.
