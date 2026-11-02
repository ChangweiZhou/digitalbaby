# Runner and admission contract

What the delivery's runner (run.py and the native modules) and its evaluator (admit.py) must emit, so that the program
checker can recompute everything that decides a verdict. Field lists, types and examples: `tools/minifly_check/SCHEMAS.md`
and `tools/minifly_check/templates/` (RECEIPT.example.json, OUTPUTS.example.jsonl, TAUGHT.example.jsonl,
ADMISSION.example.json, admit_skeleton.py). Normative text: `tools/minifly_check/SPEC.md` sections 7.4 (ADM-*, MET-*,
RUN-*) and 8. This file explains intent and the rules that are easy to get wrong; it does not restate the schemas.

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`.

The principle: the executor emits raw observations; the checker derives the measurements. Expectations for controls come
from the lock's declarations (zero_channels, declared_changed, teaching_withheld), never from the runner's own gating
code. Package B's receipt auditor derived its expected write ledger from the same `branch_allows` function that gated
the writes, so a control that performed all 288 old-relation writes, ending digest-identical to W in 832/832 receipts,
passed it (`minifly_b/REPORT.md:12-26`).

## 1. OUTPUTS rows (`minifly.outputs.v1`)

Written to `results/{admission,dev,rehearsal,official}/OUTPUTS.jsonl` (`.jsonl.gz`, or inside `RESULT_BUNDLE*.zip`).

- One row per (world, arm, pool, checkpoint, item). Every cell implied by the lock must exist with exactly
  n_items_per_pool items (LAD-01); a missing pair of arms in a world makes the verdict INCOMPLETE, never a smaller N.
- stream_hex: exactly gen_cap_bytes generated **without stopping at EOS**. The checker recomputes emitted (the bytes
  before the first eos within gen_cap) and own_eos (an eos was found) from it, and also builds the FIXED_k and
  MODE_REPEAT shortcuts from it. A stream cut at the first EOS hides what the arm would emit next and breaks both.
- emitted_hex and own_eos: what the runner itself decoded. They must equal the recomputation exactly (MET-01,
  two-source agreement). A buggy decoder shows up here (corpus fixture C01).
- prompt_hex and expected_hex: for addition formats expected_hex is recomputed from prompt_hex; a mismatch fails (MET-02).
- item_id and prompt_hex of HELD, OLD and NEW rows: exactly an item of the locked split manifest. The scorer refuses
  any other row, and LAD-01 requires every cell of those pools to hold exactly the manifest's items.
- answer_path: native, head or evaluator, as actually produced. Rows other than native never count.
- native_closure_sha, params_sha256, attempt_id: tie the row to its SOURCE_LOCK and its ATTEMPTS row (LOCK-06, RUN-01).
  Copy attempt_id from MINIFLY_ATTEMPT_ID: the attempt must be of the stage that writes that results directory, on
  that stage's worlds (CYC-01).
- No stratum field. The checker computes strata from content.

## 2. TAUGHT manifests (`minifly.taught.v1`)

`manifests/TAUGHT/<world>/<arm>.jsonl`, one per arm and world, listing what that arm was actually taught: prompt_hex,
answer_hex as presented, phase, feedback_bearing and presentations.

- The SHUF arm's manifest carries its permuted labels. ADM-SHUF computes agreement with the true answers (within
  shuf_band) and orbit consistency (both orientations of an orbit get one label) from it. A derangement on binary
  labels flips every label and fails (origin/charming-head `minifly_learned_bank_v1_20261024/cycles/CYCLE2.md:40`).
- The no_prior_teaching arm's manifest shows that the episode named in its teaching_withheld was not taught to it.
- The candidate's manifest is the ladder's training set and the base of strata and determinacy.
- Every feedback-bearing stream is declared in lock.exposure_streams. ADM-EXPO fails on any TAUGHT or stream file under
  `manifests/` or `results/` that is not declared, and on any canonical held orbit (exact prompt for E2) inside a
  feedback-bearing stream, including a calibration stream (corpus fixture C12).
- Non-feedback streams are scanned too (SPEC 20.11, checker 1.0.1): no row of any declared stream, any TAUGHT file or
  any undeclared stream-like file may carry a held item's answer together with its prompt or canonical orbit, in any
  field or encoding (text, hex, base64, byte lists, or prompt followed by answer in one value). A listening,
  presentation or warm-up stream may show held prompts without their answers. Write stream files as JSONL (JSON or
  UTF-8 text also work); a binary stream file cannot be checked and fails ADM-EXPO. Corpus S65-S67.
- Since the round-1 repair the scan also parses string values that are themselves JSON (as text, hex or base64), reads
  an answer up to 32 non-digit bytes after the prompt or orbit ('3+7 equals 10') or written before it ('10 = 3+7'),
  and reads records in stream order: a record whose value at some field ends with a held prompt or orbit, followed by
  a record whose value at the same field starts (after up to 16 non-digit bytes) with its answer, leaks it (corpus
  S84 JSON payload, S85 consecutive tape lines, S86 clean). Data files whose path names a taught, stream,
  calibration, curriculum, exposure, warm-up or listening part are stream candidates anywhere in the delivery, not
  only under manifests/ and results/. Residuals the reviewer covers: integer answers, an answer two or more records
  later or in another field of the next record, files.assets and neutrally named JSON under results/.
- G_per_sample uses the taught presentations recorded in receipts; keep the two consistent.

## 3. Receipts (`minifly.receipt.v1`)

`results/receipts/<world>/<arm>.json`, one per (world, arm) history.

- channel_digests at three moments for every state channel: birth (before the first record), post_first (right after
  the first record that the arm's manipulation affects, at the same record index in every arm of the world) and end.
- writes_by_channel: counters incremented **at the state-mutation site**, the line that actually changes the state,
  never at the decision that is supposed to allow or block it. write_counter_site names that function; ADM-CTRL-ZERO
  checks it exists in the native closure, and the reviewer reads it.
- realized_dose {metric: value}: the delivered size of the manipulation (ADM-CTRL-DOSE; a realized dose of 0.00312
  fails the 0.05 floor: corpus fixture C02b).
- records, taught_presentations, state_digest_end, native_closure_sha, params_sha256, endpoint_lock_sha256,
  attempt_id, world_role, phase.

What the checker requires, from the raw receipts (receipts win over any ADMISSION summary; disagreement is FAIL):
- ADM-CTRL-ZERO, for each arm with zero_channels Z: writes_by_channel[c] == 0 and post_first[c] == birth[c] for every c
  in Z; in W, writes on each c in Z are above 0 and post_first[c] != birth[c]. Failure: INVALID, reason
  TREATMENT_NOT_DELIVERED.
- ADM-CTRL-ISO: at post_first the arm's digests equal W's on every channel that is neither in Z nor in declared_changed.
  A control that also changes another channel (for example the stop channel; corpus fixture C03) confounds the contrast.
- ADM-CTRL-DOSE: realized_dose is at least max(min_realized_dose, CONSTRAINTS min_realized_dose).

## 4. Replay flags

The runner accepts `--only-world <W>` and `--only-arm <A>` and then reproduces exactly that history with the locked
command, writing the same rows and receipt digests. The reviewer uses this for rerun_admission (REV-02, REV-03) and, for
SEALED_CONFIRM or adoption, for the post-run replay of the (world, arm) pairs chosen by
`MFC review replay-plan <dir> --seed HEX --n 4` (REV-08 compares digests).

## 5. Write before assert

Rows and receipts are written and flushed before any assertion, sanity check or exception path runs. A failed assertion
must leave its evidence on disk, and the attempt stays counted.

## 6. Every counted run goes through `launch`

```
MFC launch --stage {testbed|technical|dev|rehearsal|official} <dir> -- <cmd ...>
```

- `launch` runs the stage's gate bundle first and refuses on a nonzero exit; it refuses a dirty or uncommitted closure,
  ENDPOINT_LOCK.json or SOURCE_LOCK.json, and dev or rehearsal attempts beyond the cycle cap.
- It measures child CPU, wall time and peak RSS, and appends an ATTEMPTS row listing every new or changed file under
  `results/` with its sha256.
- RUN-01 (enforced at T2, WARN at T1): every file under `results/{admission,dev,rehearsal,official}/` and
  `results/receipts/` appears with a matching sha256 in some ATTEMPTS row that ran from a clean closure. An official
  OUTPUTS file not covered by ATTEMPTS fails (corpus fixture C19).
- RUN-02: at most one official attempt per SOURCE_LOCK produces assessment-bearing rows, except a legal resume (6.2).
- Failed attempts stay in ATTEMPTS and count toward compute and designs_tried_k.
- `launch` re-verifies ENDPOINT_LOCK, SOURCE_LOCK and every closure sha on every invocation. A multi-phase run calls
  `launch` once per phase; never chain phases inside one unlaunched script, or a mid-run fix goes unseen.

### 6.1 Runtime closure (RUN-03)

Every counted run (technical, dev, rehearsal, official) writes `runlog/RUNTIME_CLOSURE_<attempt_id>.json` =
{schema `minifly.runtime_closure.v1`, attempt_id, files: [{path (repo-relative), sha256}]} for every repo file in
sys.modules plus the data assets it opened, from the lock's declared list (files.assets). Compute it from sys.modules,
never list it by hand: the LEARNED-BANK session hand-listed a 6-file lock while its runtime closure was 55 files,
44 of them package modules. Call the helper right after the imports and again at the very end (in a `finally`), so a
killed run still leaves its import-time closure. `launch` sets MINIFLY_REPO_ROOT, MINIFLY_DELIVERY and
MINIFLY_RUNTIME_CLOSURE and records the file in produced_paths. The same helper ships in templates/admit_skeleton.py:

```python
def write_runtime_closure(assets=()):  # stdlib only: import hashlib, json, os, sys
    root = os.path.realpath(os.environ['MINIFLY_REPO_ROOT']); base = os.path.join(root, os.environ['MINIFLY_DELIVERY'])
    paths = {os.path.realpath(m.__file__) for m in list(sys.modules.values()) if getattr(m, '__file__', None)}
    paths |= {os.path.realpath(os.path.join(base, a)) for a in assets}
    files = []
    for p in sorted(x for x in paths if x.startswith(root + os.sep) and os.path.isfile(x)):
        with open(p, 'rb') as fh:
            files.append({'path': os.path.relpath(p, root).replace(os.sep, '/'), 'sha256': hashlib.sha256(fh.read()).hexdigest()})
    out = os.path.join(base, os.environ['MINIFLY_RUNTIME_CLOSURE']); os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump({'schema': 'minifly.runtime_closure.v1', 'attempt_id': os.environ['MINIFLY_ATTEMPT_ID'], 'files': files}, fh)
    os.replace(out + '.tmp', out)
```

RUN-03 (T2 enforced, T1 WARN, like RUN-01; the checker never imports delivery code): for every counted attempt that
produced anything, the file exists, is in that attempt's produced_paths with the same sha256, and every repo `.py` file
it lists is in the SOURCE_LOCK.closure_sha256 the attempt ran under, with the same sha256. A module present at runtime
but absent from the lock FAILs as a **closure gap** (corpus S60: a `bootstrap.py` reached only by importlib). Data
assets must be declared in files.assets, files.params, endpoint_files_sha256 or SOURCE_LOCK and match. Remedy: put the
module into the static closure (or dynamic_imports) and rerun `MFC lock build <dir>`: a new design.

### 6.2 Resuming an interrupted official run (RUN-02 as amended, RUN-04)

When an official attempt is interrupted (nonzero or killed exit, an incomplete (world, arm) unit set, or a launch that
died with its container: the next `launch` records it from `runlog/INFLIGHT_<attempt_id>.json`), continue it with

```
MFC launch --stage official --resume <dir> -- <the same command>
```

- The command stays the same (CYC-03 compares its template with the rehearsal's); the resume is driven by the
  environment: MINIFLY_RESUME_OF (the interrupted attempt), MINIFLY_COMPLETED_UNITS (JSON list of [world, arm] units
  that already have official rows: skip them), MINIFLY_REPLAY_UNITS (the unit(s) launch chose at random to re-run) and
  MINIFLY_REPLAY_DIR (`results/official_replay`).
- Write the missing units' OUTPUTS rows (append to results/official/OUTPUTS.jsonl) and receipts only. Never rewrite,
  delete or duplicate a completed unit's rows, receipt or TAUGHT manifest (RUN-02). Write each unit's rows in one
  append after the unit finishes, so a kill leaves a unit complete or absent; a half-written unit can never be resumed.
- Re-run every unit in MINIFLY_REPLAY_UNITS into `results/official_replay/OUTPUTS.jsonl` (receipts, if any, under
  `results/official_replay/receipts/`). Its canonical row digest (SCHEMAS 13, attempt_id excluded) must equal the
  original's (RUN-04): bit-identical determinism under the identical lock.
- launch refuses a resume under other lock shas, after an attempt that ended normally with every unit complete, or when
  no unit is complete yet (then launch without --resume: an attempt with zero rows never blocks a new one). It records
  units (with digests), resume_of, replay_requested and replay_units in the ATTEMPTS row and exits 1 when the resume
  broke a rule; RUN-02 and RUN-04 then FAIL for good at the official and publish gates.
- CLAIM.provenance records interrupted, resume_attempts, resumed_units and replay_units (corpus S61 clean; S62 a
  rewritten unit, S63 no replay, S64 a replay that differs).

## 7. What admit.py emits

admit.py is evaluator code. It is listed in files.admit and in endpoint_files_sha256, so editing it after the lock
without an amendment fails AMD-05 (corpus fixture C15, from `native_next_v1_dev_20261023/REPORT.md:13,40`). Its
modules must not be reachable from native or runner code (LOCK-07). It runs only through `launch`.

Cycle 0 (`MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed`, then `MFC score --role admission <dir>`):
- CEILING_<cfg> and FLOOR_<cfg> arm rows in `results/admission/OUTPUTS.jsonl`, on the exact split and budget, in the
  same OUTPUTS format as learner rows.
- The configs cover CONSTRAINTS.testbed_ceiling_grid (EXACT_BP and ORACLE_FEATURES) and testbed_floor_grid
  (FROZEN_RANDOM_FEATURES, width 256/1024/4096 x readout_lr 0.001/0.005/0.02).
- The checker scores them with the delivery's own rule: at least one ceiling must PASS (ADM-TB-CEIL), and every floor
  and the best floor over the grid must FAIL (ADM-TB-FLOOR).

Technical cycle (`MFC launch --stage technical <dir> -- <cmd ...>`):
- `results/admission/FIXTURES.jsonl`:
  - directional fixtures, one per port and sign declared in lock.ports: target_margin_delta_vs_passive > 0,
    zero_coef_digest_equals_passive true, zero_replay_applications == 0 (ADM-DIR; a negative delta or a zero-coefficient
    replay with 32 applications fails, as in corpus fixture C17 from v4's reversed teaching sign);
  - their params equal the params.json values hashed in SOURCE_LOCK within the declared tolerances, and a production
    key missing from a fixture fails (ADM-FIX; v5.1's port fixture hard-coded density 0.02 against the native 0.0354,
    `PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020/AUDIT.json` finding A4);
  - gate fixtures, at least one negative (result false) and one positive (result true) per plumbing gate (ADM-GATE).
- `ADMISSION.json` (`minifly.admission.v1`), assembled from the raw rows and committed before the first dev output
  (ADM-01): both lock shas, producer {script, sha256}, the technical worlds only, the paths of the testbed outputs, the
  receipts glob and the fixtures, input_contrast, structure_meters, parent_admission_sha (ADM-REUSE) and created_utc.
  - input_contrast: metric_definition, analytic_snr, same_input_diff_noise, diff_input_same_noise, ratio,
    prompt_id_loo_acc and chance. ADM-INPUT needs ratio >= input_contrast_min (2.0 draft) and prompt_id_loo_acc >= chance
    + prompt_id_margin_min (0.10 draft). V6-R1's ratio was 0.014 (`minifly_v6_r1_dev_20261023/REVISION.md:40-52`).
  - structure_meters: shared_operand_similarity, nonshared_operand_similarity, modal_output_fraction,
    code_overlap_birth and code_overlap_end. Always reported beside distinguishability, never gated: a hash code
    passes distinguishability while sharing no structure between related operands.
- The checker computes the testbed result, taught acquisition (ADM-ACQ), SHUF validity and the controls itself from raw
  rows. ADMISSION summaries are cross-checked, never trusted.

Admission can only stop a run: INPUT_STATE_NOT_ADMITTED, NOT_ACQUIRING, TREATMENT_NOT_DELIVERED or
CANNOT_TEST_CLAIM_AS_SPECIFIED. Passing it supports no claim.

ADM-REUSE (T1 only): when the native closure, task and split equal the parent's and the parent's admission passed, cite
the parent ADMISSION sha in parent_admission_sha to skip ADM-INPUT and LOCK-12.

## 8. Checklist before the roster gate

- [ ] stream_hex is gen_cap_bytes long in every row, generated past EOS.
- [ ] TAUGHT manifest per (world, arm); every stream declared in exposure_streams.
- [ ] Receipt per (world, arm) with birth, post_first and end digests for every channel and counters at the mutation site.
- [ ] `--only-world` / `--only-arm` reproduce a history byte for byte.
- [ ] Rows written before assertions.
- [ ] Every results file produced by a `launch` attempt; one launch per phase.
- [ ] The runtime-closure helper runs after the imports and at the end of every counted run (RUN-03).
- [ ] The official runner honours MINIFLY_COMPLETED_UNITS, MINIFLY_REPLAY_UNITS and MINIFLY_REPLAY_DIR (resume).
- [ ] FIXTURES.jsonl covers every port, sign and plumbing gate; ADMISSION.json committed before any dev output.
- [ ] Then `MFC gate --stage roster <dir>`.
