# MRK-1 artifact schemas (schema v1)

Field-level reference for every artifact in SPEC Section 8, plus the kit's own files (gate
report, RULES, PROTECTED, EXPECTED, timeline fixtures). Skills link here instead of restating
fields. Where this file and `SPEC.md` disagree, the spec wins; report the mismatch.

Command prefix: `MFC` = `python3 -I tools/minifly_check/minifly_check.py`.

## 0. Conventions (apply to every schema)

| Topic | Rule |
|---|---|
| Paths | POSIX, relative, no `..`. Delivery-relative unless a field says repo-relative. Never absolute (`/Users/`, `/home/`, `/tmp/`, `C:`). |
| sha256 | Lowercase hex. A file sha is sha256 of its raw bytes as committed (so "lock sha" = sha256 of `ENDPOINT_LOCK.json` bytes). An object sha is sha256 of `canonical_json` (sorted keys, separators `,` `:`, UTF-8, no NaN). A `sha256:` prefix is tolerated and stripped. |
| UTC | Written by the kit as `YYYYMMDDTHHMMSSZ`. Readers also accept `YYYY-MM-DDTHH:MM:SSZ` and `YYYY-MM-DDTHH:MMZ`. |
| hex fields | `*_hex` = lowercase hex of raw bytes (`"3+4="` is `332b343d`). |
| Numbers | Finite only. `NaN`, `Infinity` and `-Infinity` are rejected anywhere (InvalidInput -> FAIL). `None`/`null` in a gate-relevant field is FAIL. |
| Enums | Values outside the listed enum are FAIL (SPEC 7.6). |
| `__FILL__` | Template placeholder for a value a human must decide. LOCK-01 fails while any remains. |
| `schema` | JSON documents carry a `schema` field with the name shown in their heading. JSONL rows written by the kit (`ATTEMPTS`, `CYCLES`) also carry it; runner rows (`OUTPUTS`, `TAUGHT`, `FIXTURES`) do not need it. |
| JSONL | One JSON object per line, UTF-8; blank lines ignored. Rows may be gzip-compressed (`*.jsonl.gz`) or stored as members of `RESULT_BUNDLE*.zip` (Section 8.1). |
| Integrity booleans | Never written as literals by delivery code (LOCK-08); the checker computes them. |
| World roles | `technical`, `dev`, `rehearsal`, `official`. |
| Agent families | `codex`, `claude` (session-qualified as `claude:<session>` in trailers). |

## 1. ENDPOINT_LOCK.json (`minifly.endpoint_lock.v1`)

Written by the executor via `MFC init <dir>` from `templates/ENDPOINT_LOCK.template.json`,
committed before any `results/**` exists, and changed afterwards only through `AMENDMENTS.jsonl`.

| Field | Type | Enum / notes |
|---|---|---|
| schema | str | `minifly.endpoint_lock.v1` |
| delivery | str | delivery dir name `<name>_{design\|dev}_vN_YYYYMMDD` |
| protocol_id | str | `<delivery>-p1`, bumped by AMD-02 amendments |
| created_utc | str | UTC |
| author_agent | str | `codex` \| `claude` |
| tier_declared | int | 0 \| 1 \| 2 (the computed tier is never lower, Section 4) |
| intent | str | `diagnostic` \| `promotion` (promotion = adoption/pointer/E3 promotion; forces tier 2) |
| claim_level_target | str | `E0` .. `E4` |
| task.id | str | free id |
| task.family | str | `addition_multi_digit` \| `addition_single_digit` \| `addition_component` \| `other_arithmetic` \| `non_arithmetic` |
| task.format.kind | str | `addition` \| `other` |
| task.format.sep_hex, terminator_hex, eos_hex | str | hex; prompt = a + sep + b + terminator; EOS ends the answer |
| task.format.gen_cap_bytes | int | exact length of every `stream_hex` |
| task.format.normalize_leading_zeros | bool | expected answers are normalized decimal |
| objective_distance.task_family | str | as task.family |
| objective_distance.output | str | `own_bytes_own_eos` \| `own_bytes_evaluator_length` \| `classification_label` \| `margin_or_rank` (LOCK-03 reads this as the primary's output) |
| objective_distance.held | str | `unseen_composition` \| `changed_presentation` \| `taught_only` |
| objective_distance.native_only_path | bool | |
| objective_distance.feedback | str | `gradual_warmup` \| `other` |
| objective_distance.old_protection_measured | bool | |
| units.witness | str | `pair_rectangle` \| `column_closure` |
| units.commutative | bool | |
| split.taught_manifest | str | dir holding `TAUGHT/<world>/<arm>.jsonl` |
| split.held_manifest, old_manifest, new_manifest | str | JSONL manifests (Section 5) |
| split.split_sha256, held_set_sha256 | str | sha256; the held set is fixed before any stream exists. `held_set_sha256` = sha256 of the held manifest file bytes; `split_sha256` = sha256(canonical_json({"held_manifest": <sha256 of the held file>, "old_manifest": <sha256 of the old file>, "new_manifest": <sha256 of the new file>})) (`common.split_hashes`; `MFC init` fills both when the manifests exist). AMD-05 recomputes both in the worktree and at every commit after the lock (a changed, re-pinned or deleted manifest needs an amendment); LOCK-06 requires the first TAUGHT stream commit to follow the lock commit |
| exposure_streams[] | obj | `{path_glob, feedback_bearing: bool, phase}`; every TAUGHT or stream file must match one (ADM-EXPO) |
| arms[] | obj | `{id, role, zero_channels[], declared_changed[], teaching_withheld[], min_realized_dose}` |
| arms[].role | str | `candidate` \| `no_prior_teaching` \| `lesion` \| `shuf` \| `frozen` \| `oracle_plumbing` \| `positive_control` \| `other` |
| causal_control_arm, positive_control_arm | str | arm ids |
| ports[] | obj | `{id, signs[]}`; signs are strings such as `"+"`, `"-"` |
| primary.task_id | str | |
| primary.metric | str | `exact_own_eos` \| `balanced_exact_own_eos` (anything else forces gate_type component, LOCK-03) |
| primary.unconditional | bool | must be true for an objective gate |
| primary.pool | str | `TAUGHT` \| `OLD` \| `NEW` \| `HELD` \| `PRESENTATION` |
| primary.checkpoint | str | a key of `checkpoints` (usually `final`) |
| primary.strata_counted[] | str | subset of `S0`..`S4`; E3 counts only `S2`..`S4` |
| primary.candidate_arm | str | arm id |
| primary.comparator_arms[] | str | arm ids; the comparator is max(chance, these arms, the shortcut ladder) |
| primary.ladder | str | `auto` |
| primary.chance | number | per-item chance rate |
| primary.margin_abs, absolute_floor, per_world_floor | number | >= CONSTRAINTS floors (LOCK-04) |
| primary.aggregator | obj | `{type: paired_lcb, alpha}` \| `{type: k_of_n, k}` \| `{type: all_worlds}` |
| primary.min_worlds | int | >= `min_worlds_nonpilot` unless PILOT |
| primary.n_items_per_pool | obj | `{pool: int}`; denominators come from here (LAD-01) |
| primary.rule_type | str | `declarative` \| `custom` (custom needs a UD, a non-executor reviewer and property fixtures, DEC-01) |
| gates[] | obj | plumbing gates `{id, description}`; each needs a negative and a positive fixture (ADM-GATE) |
| guards[] | obj | `{id, kind: old_drop, pool: OLD, max_drop}` and `{id, kind: old_acq, pool: OLD, margin}`; both are required whenever an OLD pool exists |
| udi | null \| obj | `{variants_manifest, alpha}` (Section 5.4) |
| checkpoints | obj | `{prefix, old_end, final}`: checkpoint name -> record index (int) at which outputs are taken |
| stage_chain | bool | true adds the `oracle_upstream` executed check (REV-02) |
| worlds.technical[], dev[], rehearsal[] | str | world ids; rehearsal worlds are fresh (CYC-03) |
| worlds.official | obj | `{mode: listed \| sealed, ids[], n}`; sealed ids are derived after GO (Section 10) |
| worlds.official_role_intended | str | `MAIN` \| `SEALED_CONFIRM` \| `PILOT` |
| schedule.paired_atomic | bool | must be true (ADM-BUDGET) |
| budget | obj | `{max_records: int, max_cpu_h: number}` |
| admission_thresholds | obj | `{analytic_input_snr: number \| "not_derivable", input_contrast_min, prompt_id_margin_min, taught_acq_margin_min, shuf_band: [lo, hi]}` |
| testbed | obj | `{ceiling_grid[], floor_grid[]}`; entries `{id, learner, ...config}` copied from CONSTRAINTS grids; arms are named `CEILING_<id>` and `FLOOR_<id>` |
| files | obj | `{admit: str, evaluator[], generators[], runner[], native[], shared[], assets[], params: str}`; delivery-relative (shared may be repo-relative) |
| endpoint_files_sha256 | obj | `{path: sha256}` of endpoint-locked files (admit.py, evaluator, generators); AMD-05 |
| lineage | obj | `{parent: "<dir>@<sha>" \| null, changes_declared[], priors_added[]}` |
| predictions[] | obj | `{id, target: primary \| guard:<id>, point, interval90: [lo, hi], p_pass, basis_findings[], base_rate_forecast: {value, source}}`; p in [0.02, 0.98]; basis ids exist in FINDINGS or are `none` |
| e3_cadence | obj | `{waiver_user_decision: UD id \| null}` |
| constraints_sha256 | str | sha256 of `PROJECT_DOCUMENTS/KB/CONSTRAINTS.json` bytes |
| user_decisions[] | str | UD ids this lock relies on |
| coprimary[] | obj | original endpoints kept after an amendment |

Example: `templates/ENDPOINT_LOCK.template.json` (every human decision is `__FILL__`).

## 2. SOURCE_LOCK.json (`minifly.source_lock.v1`)

Written by `MFC lock build <dir>` before each cycle's runs.

| Field | Type | Notes |
|---|---|---|
| schema | str | `minifly.source_lock.v1` |
| delivery | str | |
| endpoint_lock_sha256 | str | sha256 of ENDPOINT_LOCK.json |
| built_utc | str | UTC |
| checker_version | str | `X.Y.Z` of the kit that built it; the first committed SOURCE_LOCK's value drives the advisory window (Section 7.5 of the spec) |
| closure_sha256 | obj | `{repo-relative path: sha256}` of every repo file in the static import closure of files.native, files.runner and files.admit |
| native_closure_sha256 | obj | optional `{repo-relative path: sha256}` restricted to the native closure; used for lineage-family Jaccard when present (else closure_sha256 minus admit/evaluator/generators files) |
| dynamic_imports[] | str | non-literal imports, declared |
| env | obj | `{python: str, packages: {name: version}}` |
| native_closure_sha | str | sha256 over the native closure |
| runner_closure_sha | str | sha256 over the runner closure |
| params_sha256 | str | sha256 of files.params |

```json
{"schema": "minifly.source_lock.v1", "delivery": "minifly_example_dev_v1_20261101",
 "endpoint_lock_sha256": "3333...3333", "built_utc": "20261101T080000Z", "checker_version": "1.0.0",
 "closure_sha256": {"minifly_example_dev_v1_20261101/native.py": "aaaa...aaaa", "minifly_example_dev_v1_20261101/run.py": "bbbb...bbbb"},
 "dynamic_imports": [], "env": {"python": "3.9.18", "packages": {}},
 "native_closure_sha": "1111...1111", "runner_closure_sha": "5555...5555", "params_sha256": "2222...2222"}
```

## 3. AUTHORIZATION.json (`minifly.authorization.v1`)

| Field | Type | Notes |
|---|---|---|
| verbatim | str | the owner's exact words; every cap number must appear literally, else default caps apply (AUT-02) |
| utc, session, channel | str | |
| options_shown[] | obj | `{label, text, cost, objective_distance}`; no label contains `(Recommended)` (AUT-03) |
| recommendation_shown | str | separate from the options |
| stages | obj | booleans `implementation, technical, dev_cycles, official, push, push_lock, adoption, pointer_update`; `official` may instead be `{granted: bool, verbatim, utc, lock_shas_shown: [sha]}` (owner GO, Section 10 step 3) |
| caps | obj | `{max_records, max_cpu_h, max_cycles}` |
| defaults_applied | bool | |
| conditionals[] | obj | `{predicate: {artifact, pointer, op, value}, evaluated_by, result, utc}`; `artifact` is a delivery-relative path, `pointer` an RFC 6901 JSON pointer, `op` one of `eq ne lt le gt ge in exists`; otherwise `evaluated_by` names the non-beneficiary agent (AUT-04) |
| recorded_by | str | agent and session |

Default caps (CONSTRAINTS.default_grant_caps): 2 CPU-h, 5,000 records, 3 cycles; official, push,
adoption and pointer_update false. The kit's `effective_cap` uses a recorded cap only when its
number appears in `verbatim`; otherwise the default (or the recorded cap if lower).
Example: `templates/AUTHORIZATION.template.json`.

## 4. AMENDMENTS.jsonl (`minifly.amendment.v1`)

| Field | Type | Notes |
|---|---|---|
| id | str | `AMD-<n>` |
| utc | str | |
| old_lock_sha256, new_lock_sha256 | str | consecutive committed ENDPOINT_LOCK contents (AMD-01) |
| new_protocol_id | str | |
| fields_changed[] | str | dotted lock paths, e.g. `primary.margin_abs`, `guards` |
| results_seen[] | str | delivery-relative outcome files committed before the amendment (AMD-03 recomputes it): every raw outcome file under `results/` must be listed; a checker-written score file (`results/<role>/SCORES.json`, `VERDICT.json`, `TESTBED.json`) committed before the amendment may also be listed; nothing else may |
| reason | str | `ceiling` \| `testbed_invalid` \| `measurement_defect` \| `operational` |
| evidence | str | path |
| objective_distance_delta | obj | `{axis: "old->new"}` |
| original_endpoint_kept_as_coprimary | bool | |
| user_decision | str \| null | UD id (required by AMD-02 cases) |
| author_agent | str | |

A row with non-empty `results_seen` changing `primary*` or `guards*` makes the delivery tier 2.

```json
{"id": "AMD-1", "utc": "20261103T100000Z", "old_lock_sha256": "3333...", "new_lock_sha256": "6666...",
 "new_protocol_id": "minifly_example_dev_v1_20261101-p2", "fields_changed": ["testbed.ceiling_grid"],
 "results_seen": ["results/admission/OUTPUTS.jsonl"], "reason": "testbed_invalid", "evidence": "results/admission/TESTBED.json",
 "objective_distance_delta": {}, "original_endpoint_kept_as_coprimary": true, "user_decision": null, "author_agent": "codex"}
```

## 5. Manifests and params

### 5.1 params.json
Free-form object of the learner's production parameters; hashed into `SOURCE_LOCK.params_sha256`.
Example: `{"readout_lr": 0.005, "pair_density": 0.0354, "warmup_start_ceiling": "1/16"}`.

### 5.2 manifests/HELD.jsonl, OLD.jsonl, NEW.jsonl
Rows `{item_id: str, prompt_hex: str, expected_hex: str}`. `expected_hex` is recomputed by the
checker for addition formats (MET-02). Example: `{"item_id": "h001", "prompt_hex": "332b343d", "expected_hex": "37"}`.
item_id is unique within a manifest. These are the only items a HELD, OLD or NEW OUTPUTS row may carry: the scorer
refuses (InvalidInput) a row whose item_id is not in the pool's manifest or whose prompt_hex differs from it, and
LAD-01 requires every (world, arm, pool, checkpoint) cell of those pools to hold exactly the manifest's items.

### 5.3 TAUGHT rows (`minifly.taught.v1`), manifests/TAUGHT/<world>/<arm>.jsonl

| Field | Type | Notes |
|---|---|---|
| world, arm | str | |
| prompt_hex, answer_hex | str | `answer_hex` is the label the arm was taught (SHUF labels differ from truth) |
| phase | str | e.g. `teach`, `calibration` |
| feedback_bearing | bool | |
| presentations | int | |

Example: `templates/TAUGHT.example.jsonl`.

### 5.4 manifests/UDI_VARIANTS.json (UDI-01; E3 promotion only)
`{"variants": [{"base_world": str, "variant_world": str, "removed_taught_prompt_hex": [str],
"dependent_item_ids": [str], "independent_item_ids": [str]}]}`. TAUGHT(variant) must equal
TAUGHT(base) minus the removed set. For SEALED_CONFIRM the dependent half is drawn from the
seal-derived `udi` seed.

## 6. Receipt (`minifly.receipt.v1`), results/receipts/<world>/<arm>.json

Written by the runner before any assertion; the checker reads receipts raw and they take
precedence over ADMISSION summaries.

| Field | Type | Notes |
|---|---|---|
| world, world_role, arm, phase | str | |
| records | int | records processed (counts toward measured tier) |
| taught_presentations | int | denominator of G_per_sample |
| writes_by_channel | obj | `{channel: int}` incremented at the state-mutation site |
| write_counter_site | str | `<delivery-relative path>:<function qualname>` of the mutation site in the native closure |
| channel_digests | obj | `{birth: {channel: sha256}, post_first: {...}, end: {...}}` |
| state_digest_end | str | sha256 |
| realized_dose | obj | `{metric: number}` |
| native_closure_sha, params_sha256, endpoint_lock_sha256 | str | copied from the launch env (Section 12) |
| attempt_id | str | `MINIFLY_ATTEMPT_ID` |

Example: `templates/RECEIPT.example.json`.

## 7. OUTPUTS rows (`minifly.outputs.v1`)

`results/{admission,dev,rehearsal,official}/OUTPUTS.jsonl[.gz]` or inside `RESULT_BUNDLE*.zip`.

| Field | Type | Notes |
|---|---|---|
| world, world_role, arm | str | |
| pool | str | `TAUGHT` \| `OLD` \| `NEW` \| `HELD` \| `PRESENTATION` |
| checkpoint | str | a key of lock.checkpoints |
| item_id | str | |
| prompt_hex, expected_hex | str | |
| stream_hex | str | exactly `gen_cap_bytes` bytes, generated without stopping at EOS |
| emitted_hex | str | bytes before the first EOS within gen_cap (all bytes if none) |
| own_eos | bool | an EOS was emitted within gen_cap |
| answer_path | str | `native` \| `head` \| `evaluator`; only `native` rows count for learner arms. Testbed rows of `CEILING_*`/`FLOOR_*` arms carry `evaluator` and are scored only by the admission testbed (ADM-TB-*) |
| native_closure_sha, params_sha256 | str \| null | null only for evaluator-only arms |
| attempt_id | str | `MINIFLY_ATTEMPT_ID` of the producing launch |

There is no stratum field: the checker computes strata. Example: `templates/OUTPUTS.example.jsonl`.

`results/official_replay/OUTPUTS.jsonl` (SPEC 20.8) holds the determinism replay of a resumed official run: the rows of
the completed (world, arm) unit(s) the resume attempt re-ran, in the same format (world_role official). They are never
scored (MET-01 and ADM-EXPO accept the file); RUN-04 compares their canonical digest (section 13, attempt_id excluded)
with the original official rows. Only an official attempt may write under `results/official_replay/` (CYC-01), and
RUN-01 covers its files.

### 7.1 Row sources and bundles
`Ctx.iter_rows(pattern)` reads `*.jsonl`, `*.jsonl.gz` and members of `RESULT_BUNDLE*.zip`
archives anywhere in the delivery. A member's logical path is its name when that starts with
`results/` (or the zip sits at the delivery root), else `<zip dir>/<name>`. The same logical path
from two sources (for example `OUTPUTS.jsonl` and `OUTPUTS.jsonl.gz`, or a file and a zip member)
is FAIL: delete the original after compressing. RUN-01 accepts a `.gz` or zip member whose
content (decompressed) matches a launch-recorded file.

## 8. FIXTURES.jsonl (results/admission/FIXTURES.jsonl)

Directional rows: `{port, sign, target_margin_delta_vs_passive: number, zero_coef_digest_equals_passive: bool,
zero_replay_applications: int, params: {}}`. Gate rows: `{gate_id, kind: negative | positive, params: {}, result: bool}`.
Both may carry `attempt_id`. Params must equal the production params within declared tolerances (ADM-FIX).

```json
{"port": "old_relation", "sign": "+", "target_margin_delta_vs_passive": 0.031, "zero_coef_digest_equals_passive": true, "zero_replay_applications": 0, "params": {"lr": 0.1}}
{"gate_id": "stop_gate", "kind": "negative", "params": {"lr": 0.1}, "result": false}
```

## 9. ADMISSION.json (`minifly.admission.v1`)

| Field | Type | Notes |
|---|---|---|
| endpoint_lock_sha256, source_lock_sha256 | str | |
| producer | obj | `{script: "admit.py", sha256}`; script listed in endpoint_files |
| worlds[] | str | technical worlds only: the admission worlds that carry the native technical run. ADM-CTRL-ZERO/ISO/DOSE, ADM-ACQ and ADM-SHUF require receipts, TAUGHT/prefix rows and SHUF manifests there; testbed-only technical worlds (evaluator CEILING_/FLOOR_ rows) need none. ADM-01 checks that receipts cover exactly these worlds |
| testbed_outputs, fixtures | str | paths |
| receipts_glob | str | glob |
| input_contrast | obj | `{metric_definition, analytic_snr, same_input_diff_noise, diff_input_same_noise, ratio, prompt_id_loo_acc, chance}` |
| structure_meters | obj | `{shared_operand_similarity, nonshared_operand_similarity, modal_output_fraction, code_overlap_birth, code_overlap_end}` (reported, never gated) |
| parent_admission_sha | str \| null | ADM-REUSE (T1 only) |
| created_utc | str | |

The checker recomputes testbed, acquisition, SHUF and control results from raw rows.
Example: `templates/ADMISSION.example.json`; producer skeleton: `templates/admit_skeleton.py`.

## 10. ATTEMPTS.jsonl (`minifly.attempt.v1`), runlog/ATTEMPTS.jsonl

Written only by `MFC launch`, one row per attempt, also when the command fails or is interrupted.

| Field | Type | Notes |
|---|---|---|
| schema | str | `minifly.attempt.v1` |
| attempt_id | str | `ATT-<utc>-<6 hex>` |
| stage | str | `testbed` \| `technical` \| `dev` \| `rehearsal` \| `official` |
| argv | [str] | the command after `--` |
| cwd | str | repo-relative delivery dir (the command runs there) |
| utc_start, utc_end | str | |
| git_head | str | commit at launch |
| closure_clean | bool | true: launch refuses otherwise |
| closure_clean_after | bool | closure still committed and unmodified after the run; false makes RUN-01 reject the outputs |
| endpoint_lock_sha256, source_lock_sha256 | str \| null | source_lock null only at testbed |
| native_closure_sha, params_sha256 | str \| null | from SOURCE_LOCK (designs_tried_k) |
| exit_code | int | child exit code (negative = signal; 127 = could not start) |
| wall_s | number | |
| cpu_children_user_s, cpu_children_sys_s | number | RUSAGE_CHILDREN deltas |
| max_rss_kb | int | peak RSS of the child tree |
| produced_paths[] | obj | `{path, sha256}`: new or changed files under `results/` and `manifests/TAUGHT/` (the runner writes TAUGHT manifests per arm; seal-derived worlds can only get them during the run) and the runtime closure `runlog/RUNTIME_CLOSURE_<attempt_id>.json` (section 10.1), delivery-relative. Stage scope (CYC-01): see below |
| removed_paths[] | str | files under results/ that disappeared |
| gate_report | str | the gate report the launch ran (`checks/GATE_<bundle>_<utc>.json`) |
| launch_error, snapshot_error, interrupted, closure_problems_after | optional | diagnostics |
| units[] | obj | official attempts: `{world, arm, n_rows, digest}` of every (world, arm) unit the attempt wrote into results/official/OUTPUTS rows (canonical digest of section 13, attempt_id excluded), computed by launch when the command ends; a later resume must leave them unchanged (RUN-02) |
| resume_of, replay_requested[], replay_units[] | optional | a `launch --stage official --resume` attempt: the interrupted attempt it continues, the [world, arm] unit(s) launch chose at random to replay, and the units the replay rows hold (section 10.3) |
| recovered, rusage_lost | optional | a row written from an in-flight marker (section 10.2): `{utc, reason}`; exit_code -1, interrupted true, CPU and wall time 0 (lost) |
| units_error | optional | why `units` could not be computed (RUN-02 then fails closed for a resume chain) |

```json
{"schema": "minifly.attempt.v1", "attempt_id": "ATT-20261102T101500Z-a1b2c3", "stage": "dev",
 "argv": ["python3", "-I", "run.py", "--worlds", "w_dev_1"], "cwd": "minifly_example_dev_v1_20261101",
 "utc_start": "20261102T101500Z", "utc_end": "20261102T104012Z", "git_head": "9f3c...", "closure_clean": true,
 "closure_clean_after": true, "endpoint_lock_sha256": "3333...", "source_lock_sha256": "4444...",
 "native_closure_sha": "1111...", "params_sha256": "2222...", "exit_code": 0, "wall_s": 1512.4,
 "cpu_children_user_s": 1498.2, "cpu_children_sys_s": 3.1, "max_rss_kb": 512004,
 "produced_paths": [{"path": "results/dev/OUTPUTS.jsonl", "sha256": "7777..."}], "removed_paths": [],
 "gate_report": "checks/GATE_roster_20261102T101458Z.json"}
```

Stage scope (CYC-01 at every tier; `launch` exits 1 on an attempt that breaks it and refuses beforehand a command
line naming a lock world of another stage). Each stage writes only its own results directory and its own worlds:

| stage | results directory | worlds (receipts, TAUGHT, OUTPUTS rows) | rows |
|---|---|---|---|
| testbed | `results/admission/` | technical | CEILING_*/FLOOR_* arms only; receipts only for those arms; TAUGHT of any lock world only while no learner attempt precedes it (cycle 0 stream generation), afterwards only CEILING_*/FLOOR_* manifests of technical worlds |
| technical | `results/admission/` | technical | any arm |
| dev | `results/dev/` | dev | any arm |
| rehearsal | `results/rehearsal/` | rehearsal | any arm |
| official | `results/official/` and (resume replay) `results/official_replay/` | official (listed ids, else any world no other role lists) | any arm |

A world list the lock leaves empty admits any world that no other role lists. Every OUTPUTS row under
`results/<role>/` whose `attempt_id` names an ATTEMPTS row must name one of the matching stage, on a world of that
stage. A row whose `attempt_id` names no ATTEMPTS row is RUN-01's (T2 enforced, T1 WARN), and for official rows also
RUN-02's. A RESULT_BUNDLE*.zip outside `results/<role>/` is judged by its members' logical paths.

### 10.1 Runtime closure (`minifly.runtime_closure.v1`), runlog/RUNTIME_CLOSURE_<attempt_id>.json

Written by the runner of every counted run (stage technical, dev, rehearsal or official) at its end, with the
runner-contract helper (templates/admit_skeleton.py `write_runtime_closure`; minifly-e3-design
ADMISSION_AND_RUNNER_CONTRACT.md 6.1): call it right after the imports and again at the very end.

| Field | Type | Notes |
|---|---|---|
| schema | str | `minifly.runtime_closure.v1` |
| attempt_id | str | `MINIFLY_ATTEMPT_ID` (the file name carries it too) |
| files[] | obj | `{path, sha256}`: path repo-relative (POSIX), for every repo file in sys.modules at the end of the run plus the data assets the run opened from its declared list (files.assets); computed, never listed by hand |

RUN-03 (T2 enforced, T1 WARN): for every counted attempt that produced any path, the file is in its produced_paths with
the current sha256, every listed repo `.py` file is in the SOURCE_LOCK.closure_sha256 the attempt ran under (found by
its source_lock_sha256 in the worktree or history) with the same sha256 (a file present at runtime but absent from the
lock is a **closure gap**), and every other listed file is declared in files.assets, files.params,
endpoint_files_sha256 or SOURCE_LOCK with the same sha256 (a files.assets entry without a locked sha is compared with
its bytes at the attempt's git_head). An attempt may not produce another attempt's runtime closure.

```json
{"schema": "minifly.runtime_closure.v1", "attempt_id": "ATT-20261102T101500Z-a1b2c3",
 "files": [{"path": "minifly_example_dev_v1_20261101/native.py", "sha256": "aaaa...aaaa"},
           {"path": "minifly_example_dev_v1_20261101/run.py", "sha256": "bbbb...bbbb"}]}
```

### 10.2 In-flight marker (`minifly.inflight.v1`), runlog/INFLIGHT_<attempt_id>.json

`launch` writes it before it starts the command and deletes it once the ATTEMPTS row is appended. It holds the fields
the ATTEMPTS row would take from the launch (attempt_id, stage, argv, cwd, utc_start, git_head, both lock shas,
native_closure_sha, params_sha256, gate_report, resume fields), the launch `pid`, `boot_id` and `host`, and `before`
(the produced-paths snapshot taken before the command). When the launch process itself died (container restart), the
next `launch` or `cycle close` turns a marker whose process is gone into an ATTEMPTS row (exit_code -1, interrupted,
recovered, rusage_lost; produced_paths against `before`; official units computed from the rows on disk) and deletes the
marker; a marker whose launch is still alive refuses the new launch. RUN-01 FAILs while a marker without an ATTEMPTS
row exists. Never commit a marker.

### 10.3 Resuming an interrupted official run (RUN-02 as amended, RUN-04)

`MFC launch --stage official --resume <dir> -- <the same command>`. A resume attempt (resume_of set) is legal only if
(a) its ENDPOINT_LOCK and SOURCE_LOCK shas equal those of the interrupted attempt and of every earlier official
attempt; (b) it resumes the latest earlier official attempt, whose ATTEMPTS row exists and which was interrupted
(nonzero or killed exit, recovered, or an incomplete (world, arm) unit set against the listed official worlds (else the
worlds seen) times the lock's learner arms, or fewer worlds than official.n); (c) its official rows are only for units
no earlier official attempt has rows for, every earlier attempt's recorded `units` still hash to their digests, it
rewrote or removed no other file an earlier official attempt produced (receipts, TAUGHT manifests), and each official
receipt names the attempt whose rows cover its unit; and (d, RUN-04) if it wrote official rows, it re-ran every unit in
replay_requested (at least one completed unit) into `results/official_replay/OUTPUTS.jsonl` with a canonical digest
equal to the original's. `launch` refuses a resume that cannot be legal (other lock shas, nothing interrupted, no
completed unit yet: then launch without --resume) and exits 1 after one that broke a rule. Without --resume, an official
attempt after an assessment-bearing one FAILs RUN-02 as before.

## 11. CYCLES.jsonl (`minifly.cycle.v1`), runlog/CYCLES.jsonl

Appended by `MFC cycle close <dir> --kind K --decision D --reason TEXT`.

| Field | Type | Notes |
|---|---|---|
| k | int | cycle number (1-based, all kinds) |
| kind | str | `testbed` \| `technical` \| `dev` \| `rehearsal` (the last three count against `caps.max_cycles`) |
| decision | str | `continue` \| `revise` \| `stop` \| `escalate` |
| reason | str | one sentence |
| utc | str | |
| computed.native_closure_sha, params_sha256, source_lock_sha256 | str \| null | current SOURCE_LOCK |
| computed.attempts[] | str | ids of the attempts of this kind that no earlier close covers (`cycle close` refuses to cover another stage's attempts) |
| computed.latest_gate | obj \| null | `{path, stage, dry, verdict, utc, source_lock_sha256, sha256}` |
| computed.outcomes_readable | bool | any learner-arm OUTPUTS row or receipt exists |
| computed.designs_tried_k | int | distinct (native_closure_sha, params_sha256) pairs over learner outcome rows, receipts and outcome-bearing attempts |

```json
{"schema": "minifly.cycle.v1", "k": 3, "kind": "dev", "decision": "revise", "reason": "Candidate below FIXED_1 on dev; widen pair memory.",
 "utc": "20261102T120000Z", "computed": {"native_closure_sha": "1111...", "params_sha256": "2222...", "source_lock_sha256": "4444...",
 "attempts": ["ATT-20261102T101500Z-a1b2c3"], "latest_gate": {"path": "checks/GATE_roster_20261102T101458Z.json", "stage": "roster",
 "dry": false, "verdict": "PASS", "utc": "20261102T101458Z", "source_lock_sha256": "4444...", "sha256": "9999..."},
 "outcomes_readable": true, "designs_tried_k": 2}}
```

Accounting (CYC-01, and the `launch` cap, which refuses technical, dev and rehearsal attempts beyond it): every
ATTEMPTS row of stage testbed, technical, dev or rehearsal belongs to
at most one CYCLES row of the same kind, and every listed id exists in ATTEMPTS. Attempts no row covers form one open
cycle per stage; closed test cycles plus open test cycles are at most `caps.max_cycles`; at most one stage is open at a
time (`launch` refuses another stage's attempt while one is open); at the non-dry official stage (gate or launch) and
once an official attempt exists, no test attempt may be uncovered. CYCLES.jsonl and ATTEMPTS.jsonl are append-only:
every committed row survives in every later commit and in the worktree (a shallow clone gives MISSING).

Exit criteria (CYC-02, and `cycle close` refuses `continue` without them): testbed - a
`TESTBED.json` under results/; technical - a PASS non-dry roster gate report at the cycle's
SOURCE_LOCK dated no later than the close; dev - `results/dev/SCORES.json`; rehearsal - a PASS
`DRY_official_*` report dated no later than the close.

## 12. Runner environment (set by launch)

`MINIFLY_ATTEMPT_ID`, `MINIFLY_STAGE`, `MINIFLY_DELIVERY` (repo-relative), `MINIFLY_ENDPOINT_LOCK_SHA256`,
`MINIFLY_SOURCE_LOCK_SHA256`, `MINIFLY_NATIVE_CLOSURE_SHA`, `MINIFLY_PARAMS_SHA256`, `MINIFLY_REPO_ROOT` (absolute repo
root) and `MINIFLY_RUNTIME_CLOSURE` (delivery-relative path of the runtime closure to write, section 10.1). The command
runs with cwd = the delivery dir and inherited stdio. Copy these values into rows and receipts.

A resume (`launch --stage official --resume`, section 10.3) adds `MINIFLY_RESUME_OF` (the interrupted attempt id),
`MINIFLY_COMPLETED_UNITS` (JSON list of [world, arm] units that already have official rows: skip them),
`MINIFLY_REPLAY_UNITS` (JSON list of the [world, arm] unit(s) launch chose at random to re-run) and `MINIFLY_REPLAY_DIR`
(`results/official_replay`). The command line is the same as the interrupted attempt's (CYC-03 compares templates).

## 13. REVIEW.json (`minifly.review.v1`), review/REVIEW.json

| Field | Type | Notes |
|---|---|---|
| delivery, reviewer_agent, reviewer_session, reviewer_branch, author_agent | str | reviewer family differs from the author (REV-01) |
| reviewed_endpoint_lock_sha256, admission_sha256 | str | WIT-01 |
| utc | str | |
| executed_checks[] | obj | `{id, kind, script, script_sha256, cmd, output, output_sha256, cpu_cap_s, result, compared_to, match}`; kinds `rerun_admission`, `reject_input`, `trace_number`, `shortcut_attempt`, `oracle_upstream`, `replay`; scripts and outputs committed under `review/`. REV-02 requires compared_to and a boolean match for rerun_admission; for other kinds a match that is present must be a boolean |
| canary | obj | `{seed, n, detected, report}`; report = `checks/CANARY_<utc>.json` |
| cheapest_shortcut_attempt | obj | `{description, script, passed_locked_rule: bool}`; true forces BLOCK |
| fm_judgments | obj | `{FM-xx: text}` |
| findings[] | obj | `{id, fm_ids[], claim, severity: BLOCKER \| MAJOR \| MINOR \| NOTE, executed: bool, status: OPEN \| CONFIRMED \| CONFIRMED_WITH_CORRECTION \| REFUTED \| NEW_DEFECT \| SCOPE_LIMITED, reconciled_by_author: {utc, note, evidence}, countersigned_by_reviewer: bool, ruling_user_decision}`; a finding without an executed check is UNVERIFIED |
| source_lock_acks[] | obj | `{sha, utc, note}` |
| seal | obj \| null | `{nonce_sha256, revealed_nonce, reveal_utc}` |
| postrun | obj \| null | `{replay: [{world, arm, expected_digest, observed_digest}], utc[, seed]}`; digests per the replay digest below |
| decision | str | `ADMIT` \| `BLOCK` |

Example: `templates/REVIEW.template.json`.

Seal files (Section 10 of the spec): `review/SEAL_COMMIT.json` = `{schema: minifly.seal_commit.v1,
endpoint_lock_sha256, source_lock_sha256, nonce_sha256, reviewer_agent, reviewer_session, utc}`;
`review/SEAL_REVEAL.json` = `{schema: minifly.seal_reveal.v1, nonce, utc}`.

Seal derivation (spec Section 10 step 5): the nonce is the `secrets.token_hex(32)` text; key =
sha256(nonce UTF-8 || GO verbatim UTF-8) with GO verbatim = AUTHORIZATION.stages.official.verbatim;
world_seed_i = int.from_bytes(HMAC-SHA256(key, "L1|L2|confirm_worlds|i")[:8], 'big') with L1/L2 the current
ENDPOINT_LOCK and SOURCE_LOCK shas; **official world id i = `w_seal_` + the 16 lowercase hex digits of
world_seed_i** (`'w_seal_%016x' % seed`), for i < worlds.official.n. The `udi` and `replay` purposes use i = 0.

Replay digest (REV-08, `MFC review replay-plan`): for one (world, arm), sha256 over the canonical JSON
(`canonical_json`) of each official OUTPUTS row restricted to `world, world_role, arm, pool, checkpoint, item_id,
prompt_hex, expected_hex, stream_hex, emitted_hex, own_eos, answer_path, native_closure_sha, params_sha256`
(attempt_id excluded: it differs between attempts), rows sorted by (pool, checkpoint, item_id), each followed by
`\n`. expected_digest and observed_digest must both equal the checker's digest of the official rows.

Canary report `checks/CANARY_<utc>.json` (`minifly.canary_report.v1`, written by `MFC canary`): `schema, delivery,
utc, seed, n_requested, n_mutants, detected, checker_sha256, checker_version, git_mode, git_head,
endpoint_lock_sha256, source_lock_sha256, tier, decisive_role, applicable_fraction, operators[], reproduction[],
mutants[], copied_outside_delivery, pointer_copy, copies_kept, error, results[] (CAN-01..03), exit_code, verdict`.
REV-05 requires REVIEW.canary `{seed, n, detected}` to equal the report's seed, n_mutants and detected, verdict
PASS with CAN-01..03 PASS, the current ENDPOINT_LOCK sha, and the report committed after the lock.

## 14. SCORES.json and VERDICT.json (written by `MFC score`)

`results/<role>/SCORES.json`: `cells [{world, world_role, arm, pool, checkpoint, stratum, determined, n, n_exact}]`,
`ladder {"<world>|<pool>|<checkpoint>": {member: value}}`, `best_shortcut {name, value}`, `strata {stratum: count}`,
`determinacy {determined, total}`, `modal_fraction`, `entropy_bits`, `G`, `G_per_sample`, `G_per_cpu_h`,
`old {end, final, drop, acq}`, `udi`, `designs_tried_k`, `inert_alarms []`, `met_mismatches []`,
`comparators {arm: mean counted rate}` (the declared comparator_arms; LAD-03), `decision_comparators {arm: rate}`
(the decision comparator of SPEC 20.6: comparator_arms plus every declared no_prior_teaching, shuf, frozen and lesion
arm; the headline names any of them at or above the candidate), `best_other_arm {name, value}`.
`results/admission/TESTBED.json` (score --role admission) holds the ceiling/floor results.

`results/official/VERDICT.json`: `verdict` (`PASS` \| `FAIL` \| `INVALID` \| `INCOMPLETE` \| `DEV_NOT_EVIDENCE` \|
`CANNOT_TEST_CLAIM_AS_SPECIFIED`), `invalid_reason` (`INTEGRITY_ALARM` \| `TREATMENT_NOT_DELIVERED` \| `NONFINITE` \| null),
`shadow_verdict`, `primary {mean_cand, mean_comp, per_world [], per_world_detail [{world, v_cand, v_comp, comparator,
...}], lcb, method, comparator_arms [], implied_comparator_arms []}`, `gates {}`, `guards {}`, `decide_version`,
`endpoint_lock_sha256`, `source_lock_sha256`, `cells_sha256`, `world_roles {}`. `primary.comparator_arms` is the decision
comparator (SPEC 9.5 as amended in 20.6): the declared comparator_arms, then every other declared arm whose role is
no_prior_teaching, shuf, frozen or lesion (`implied_comparator_arms`; never oracle_plumbing, positive_control or
CEILING_*/FLOOR_* arms; none in the cycle-0 testbed view). Every one of them needs its primary cells (else INCOMPLETE).
INERT-01's zero-variance contrast reads the declared comparator_arms only.

SCORES, VERDICT and TESTBED directly under `results/<role>/` are checker-written and exempt from RUN-01.

Cells implied by the lock (LAD-01): for every world of the role and every lock arm, the (pool, checkpoint) cells
the rule, guards and gates read plus the fixed measurement points, namely the primary `(pool, checkpoint)`; OLD
at `old_end` and `final` (OLD guards, old_end/old_final); TAUGHT at `prefix` (acquisition, ADM-ACQ); HELD and NEW
at `final` (G, new_final); any gate's `(pool, checkpoint)`; and every other declared pool at `final`. Only pools in
`primary.n_items_per_pool` and keys of `checkpoints` are ever implied. Each implied cell has exactly
n_items_per_pool items. A cell with an undeclared world, arm, pool or checkpoint FAILs; an extra cell of declared
pools and checkpoints is allowed only when it is complete too.

```json
{"cells": [{"world": "w_dev_1", "world_role": "dev", "arm": "W", "pool": "HELD", "checkpoint": "final", "stratum": "S3", "determined": true, "n": 12, "n_exact": 5}],
 "ladder": {"w_dev_1|HELD|final": {"CONST": 0.05, "FIXED_1": 0.11, "CANON": 0.08}}, "best_shortcut": {"name": "FIXED_1", "value": 0.11},
 "strata": {"S0": 4, "S2": 20, "S3": 24, "S4": 12}, "determinacy": {"determined": 48, "total": 56}, "modal_fraction": 0.21, "entropy_bits": 2.9,
 "G": -0.002, "G_per_sample": -0.0001, "G_per_cpu_h": -0.0007, "old": {"end": 0.9, "final": 0.88, "drop": 0.02, "acq": 0.31},
 "udi": null, "designs_tried_k": 2, "inert_alarms": [], "met_mismatches": []}
```

```json
{"verdict": "FAIL", "invalid_reason": null, "shadow_verdict": null,
 "primary": {"mean_cand": 0.11, "mean_comp": 0.106, "per_world": [0.01, -0.02], "lcb": -0.04, "method": "paired_lcb"},
 "gates": {}, "guards": {"old_drop": "PASS", "old_acq": "FAIL"}, "decide_version": "1.0.0",
 "endpoint_lock_sha256": "3333...", "source_lock_sha256": "4444...", "cells_sha256": "8888..."}
```

## 15. CLAIM.json (`minifly.claim.v1`)

Generated by `MFC claim`; agents edit only `findings_proposed`, `findings_none_reason` and `notes`.

| Group | Fields |
|---|---|
| levels and gates | delivery, protocol_id, level_attempted, level_earned (`E0`..`E4` \| `NONE`), reuse_specificity (`not_tested` \| `udi_pass` \| `udi_fail`), primary_gate (`PASS` \| `FAIL` \| `INVALID` \| `INCOMPLETE` \| `NOT_OPENED` \| `CANNOT_TEST_CLAIM_AS_SPECIFIED`), dev_shadow_verdict, invalid_reason, stop_reason, gate_type (`objective` \| `component`), labels[] (`integrated_intervention` \| `weak_comparator` \| `priors:<x>`), worlds_role (`DEV` \| `PILOT` \| `MAIN` \| `SEALED_CONFIRM`), n_worlds |
| objective_metric | held_exact_own_eos (number \| `not_measured`), held_by_stratum, determined_fraction, old_end, old_final, old_drop, new_final, best_shortcut {name, value}, best_other_arm {name, value}, headroom, G, G_per_sample, G_per_cpu_h |
| provenance | designs_tried_k, amendments[], coprimary_results[], posthoc_files[], lineage_parent, family_id, native_path_only, compute {records_total_incl_failed, cpu_h_total_incl_failed, family_cpu_h}, provenance {interrupted (bool: an official attempt exited nonzero, was killed or was recovered from its in-flight marker), resume_attempts [attempt_id], resumed_units [[world, arm]] (official units written by resume attempts), replay_units [[world, arm]] (units they re-ran into results/official_replay/)} (SPEC 20.8), verdict_source {decide_version, verdict_sha256, endpoint_lock_sha256, source_lock_sha256} |
| knowledge and review | predictions_scored [{id, observed, hit, interval_score, brier}], findings_proposed[], findings_none_reason, review {path, decision}, review_debt_id, authorization {path, recommendation_shown}, confirm_attempt {family_id, index, alpha}, notes |
| identity | checker_sha256, headline |

Headline grammar: `<PRIMARY_GATE> | earned <E> (attempted <E>) | held exact own-EOS <x>% [S2+ determined, n=<n>] vs best shortcut <name> <y>% | G <g>pp | OLD final <o>% (drop <d>pp) | <n> <ROLE> worlds | best of k=<k>`,
optional suffixes ` | <arm> >= candidate`, ` | amended: co-primary <gate>`, ` | priors: <list>`, ` | WEAK_COMPARATOR`,
` | dev shadow <verdict>`. The first non-empty line of REPORT.md equals the headline (CLM-05).
Example: `templates/CLAIM.example.json`.

## 16. Pointer (`minifly.pointer.v1`)

`{schema, track, delivery, claim, claim_sha256, level_earned, primary_gate, gate_type, worlds_role, objective_metric,
narrowing_rows[], updated_utc, previous_snapshot, user_decision}`. Fields are byte copies of the referenced CLAIM
(PTR-02); no bare boolean keys matching `confirm|pass|passes|passed|success|qualified` at any depth (PTR-03); the
previous pointer is snapshotted into `<TRACK>_POINTER_HISTORY/` in the same commit. Example: `templates/POINTER.example.json`.

## 17. ERRATA.jsonl

`{id, utc, old_primary_gate, new_primary_gate, classification: BUG_FIX | POST_HOC, diff_summary, fixture_id, reviewer_signoff}`.
POST_HOC never upgrades a verdict; a BUG_FIX upgrade needs a corpus fixture id and a reviewer sign-off.
Example: `{"id": "ERR-1", "utc": "20261110T090000Z", "old_primary_gate": "PASS", "new_primary_gate": "FAIL", "classification": "BUG_FIX", "diff_summary": "MET-02 carry bug in expected_hex", "fixture_id": "C01", "reviewer_signoff": "review/ERRATA_SIGNOFF.json"}`.

## 18. KB ledgers (PROJECT_DOCUMENTS/KB/*.jsonl; append-only, `merge=union`)

| Ledger | Fields | Duplicate key (same key with different content is FAIL; identical lines dedupe) |
|---|---|---|
| USER_DECISIONS | id `UD-<yyyymmdd>-<n>`, utc, verbatim, session, channel, scope[] (e.g. `amendment:<delivery>`, `e3_waiver:<delivery>`, `constraint:<key>`), numbers_restated[], recorded_by | id |
| FINDINGS | id `F-<yyyymmdd>-<agent>-<n>` or seed `F-UF-xx`/`F-ARCH-nn`, kind (`finding` \| `activation` \| `narrowing` \| `supersession` \| `retraction` \| `transfer` \| `reconciliation`), refers_to, claim, level, conditions {substrate, input_code, priors_supplied, feedback, dose, comparator, best_shortcut, worlds, metric_definition}, effect {value, ci, unit}, source `path:line@sha`, status, delivery, author_agent, utc, caveat | id |
| INCIDENTS | id, kind (`report` \| `closure`), fm_class, delivery, symptom, evidence, found_by_agent, found_utc, caught_at_discovery, checker_sha_at_discovery, heldout_eligible, check_id, fixture_id, status | (id, kind) |
| DELIVERY_LEDGER | delivery, utc, legacy, task_family, level_earned, primary_gate, gate_type, worlds_role, opened_held_E3 (`true` \| `false` \| `"unknown"`), e3_basis[], objective_metric {held, old_final, G}, family_id, claim_sha256 | none (one row per `claim` run) |
| CONFIRM_LEDGER | family_id, delivery, attempt_index, alpha_allocated, seal_reveal_commit, utc | (family_id, attempt_index) |
| REVIEW_DEBT | id, delivery, utc, reason, recorded_by, repaid_by | (id, open \| repaid): repayment is a new row with the same id and repaid_by set |
| CANARY_COMMITMENTS | batch_id, author_identity, utc, bundle_sha256, key_sha256, n, n_clean, fm_coverage, checker_sha_under_test, revealed_utc | (batch_id, committed \| revealed) |

Finding status is a fold over the finding row and rows with `refers_to` = its id: the most
restrictive of proposed/provisional < active < unverifiable_in_repo < narrowed < superseded <
retracted (order-independent, so union merges are safe). R-levels are computed by `kb.py`.

Examples (one line each):

```json
{"id": "UD-20261101-1", "utc": "20261101T090000Z", "verbatim": "Freeze checker 1.0.0 and CONSTRAINTS v1 with margin 0.05 ...", "session": "owner-chat-2026-11-01", "channel": "chat", "scope": ["constraint:threshold_floors"], "numbers_restated": ["0.05"], "recorded_by": "claude:session_x"}
{"id": "I-0007", "kind": "report", "fm_class": "FM-02", "delivery": "minifly_b", "symptom": "lesion arm digests equal W", "evidence": "minifly_b/REPORT.md:12-26@f92365e4", "found_by_agent": "claude", "found_utc": "20261024T061226Z", "caught_at_discovery": false, "checker_sha_at_discovery": null, "heldout_eligible": false, "check_id": null, "fixture_id": null, "status": "open"}
{"delivery": "minifly_example_dev_v1_20261101", "utc": "20261105T120000Z", "legacy": false, "task_family": "addition_single_digit", "level_earned": "E3", "primary_gate": "PASS", "gate_type": "objective", "worlds_role": "MAIN", "opened_held_E3": true, "e3_basis": ["pool HELD", "strata S2-S4", "ADM-EXPO PASS"], "objective_metric": {"held": 0.625, "old_final": 0.91, "G": 0.413}, "family_id": "FAM-1a2b3c4d", "claim_sha256": "8888..."}
{"family_id": "FAM-1a2b3c4d", "delivery": "minifly_example_dev_v1_20261101", "attempt_index": 0, "alpha_allocated": 0.05, "seal_reveal_commit": "abcd...", "utc": "20261104T100000Z"}
{"id": "DEBT-20261102-1", "delivery": "minifly_example_dev_v1_20261101", "utc": "20261102T090000Z", "reason": "codex unavailable", "recorded_by": "claude:session_x", "repaid_by": null}
{"batch_id": "HB-2026-11", "author_identity": "owner", "utc": "20261103T080000Z", "bundle_sha256": "aaaa...", "key_sha256": "bbbb...", "n": 10, "n_clean": 3, "fm_coverage": ["FM-02", "FM-06"], "checker_sha_under_test": "cccc...", "revealed_utc": null}
```

```json
{"id": "F-20261105-codex-1", "kind": "finding", "refers_to": null, "claim": "...", "level": "E1", "conditions": {"substrate": "...", "best_shortcut": "FIXED_1"}, "effect": {"value": 0.04, "ci": [0.01, 0.07], "unit": "abs"}, "source": "minifly_example_dev_v1_20261101/REPORT.md:12@9f3c...", "status": "proposed", "delivery": "minifly_example_dev_v1_20261101", "author_agent": "codex", "utc": "20261105T120000Z", "caveat": null}
```

`MFC kb score` output (`minifly.kb_score.v1`): `families {task_family: {n_scored_deliveries, n_pass,
laplace_base_rate, n_primary_predictions, brier, brier_base, brier_skill, n_interval_predictions, hit_rate,
label INSUFFICIENT_N | DEGENERATE_BASE | OK}}`, program-level `n_primary_predictions`, `brier_mean` (6 decimals)
and `skill` (one `<family>: <label> (n=<n>, brier_skill <v>)` text per family), `finding_transfers`,
`excluded [{delivery, reason}]`. Only the latest DELIVERY_LEDGER row per delivery whose `claim_sha256` equals the
CLAIM.json bytes is scored. A `predictions_scored` entry in the SPEC 8 form `{id, observed, hit, interval_score,
brier}` takes `target` and `p_pass` from the lock prediction with that id; an entry is scored when it has a
numeric `observed` or a boolean `hit` (or `status: scored`).

Governance review record `PROJECT_DOCUMENTS/KB/GOVERNANCE/REVIEW_<id>.json` (`minifly.governance_review.v1`, GOV-REV):
`{schema, id, reviewer_agent, reviewer_session, author_agent, proposal, decision APPROVE, utc[, base, head |
reviewed_commit, paths[], findings[]]}`, cited by a `Governance-Review: <path or id>` trailer in the range. It binds
to the range when it names the range base, a commit in the range, or paths covering every changed protected
path; a record that names none of these binds when its introducing commit lies inside the range. Its
introducing commit carries only the reviewer family's Acting-Agent and sits in a `<remote>/<family>/*` branch.

### 18.1 CONSTRAINTS.json and FAILURE_CLASSES.json
CONSTRAINTS keys are listed in SPEC 13.1 (identity, native_forbidden {imports, calls, identifiers},
supplied_arithmetic_heuristics, evaluator_only_exemptions, io, warmup, objective_targets,
threshold_floors, tier_thresholds {t0_max_cpu_h, t1_max_cpu_h, t1_max_records, family_t1_max_cpu_h},
default_grant_caps, e3_cadence_max, e3_cadence_counts_legacy, review_debt_max_open,
required_arms_by_level, testbed_floor_grid and testbed_ceiling_grid (entries `{id, learner, ...}`),
confirm_alpha_spending, verification_subset_max_mb). Only the owner changes it (UD with scope
`constraint:<key>`). FAILURE_CLASSES.json = `{FM-xx: {title, definition, check_ids[], fixture_ids[],
judgment_question, archive_fc[], incidents[]}}`, e.g. `{"FM-02": {"title": "undelivered manipulation",
"definition": "...", "check_ids": ["ADM-CTRL-ZERO", "ADM-CTRL-DOSE", "INERT-01"], "fixture_ids": ["C02a", "C02b"],
"judgment_question": "Does the write counter sit at the mutation site?", "archive_fc": ["FC01"], "incidents": ["I-0007"]}}`.

## 19. Kit files (protected)

### 19.1 RULES.json (`minifly.rules.v1`)
`{schema, checker_version, notes[], rules: [entry], bundles: {name: bundle}}`.

Entry: `{id, module, title, stages[], tier_min: 0|1|2, introduced_in: "X.Y.Z", status, fm_ids[], applies_when, warn_below_tier?}`.
- `status`: `enforced` \| `advisory` (FAIL/MISSING reported as WARN) \| `planned` (registered, not implemented,
  excluded from every bundle and listed in README "Not yet enforced"; a bundle naming it is a CHECKER_ERROR).
- `applies_when`: text starting with `always` means NOT_APPLICABLE is never allowed; any other text declares the
  condition under which a check may return NOT_APPLICABLE (otherwise NOT_APPLICABLE becomes FAIL).
- `warn_below_tier: N`: FAIL/MISSING become WARN while the computed tier is below N (RUN-01, CLM-07).
- `tier_min`: below it the check is NOT_APPLICABLE.
- Advisory window: when `introduced_in` is later than the `checker_version` of the delivery's first committed
  SOURCE_LOCK (a genuine version between 1.0.0 and the running checker), FAIL/MISSING become ADVISORY_FAIL, except
  at tier-2 official, publish and pointer stages. SELF-01, SCHEMA-01 and TIER-01 are never downgraded.

Example entry: `{"id": "RUN-01", "module": "launch", "title": "...", "stages": ["roster", "official", "publish"],
"tier_min": 0, "introduced_in": "1.0.0", "status": "enforced", "fm_ids": ["FM-13", "FM-19"],
"applies_when": "always (FAIL/MISSING become WARN below tier 2)", "warn_below_tier": 2}`.

Bundle: `{description, extends?, kind?, checks: [entry]}` where an entry is an id, `{id, when}` or
`{group, when, primary[], fallback[]}`; `when` is one of `t2`, `sealed_confirm`, `sealed_or_adoption`,
`oc_intended`, `udi_applicable`. A group runs `primary`; if any primary result blocks it runs `fallback`,
and when every fallback result is non-blocking with at least one PASS, the blocking primary results are
reported as WARN (used for REV-01..07 or DEBT-01). Bundles with `kind: command` (canary, governance,
heldout, lint) list what those commands run and are not gate stages.

### 19.2 PROTECTED.json (`minifly.protected.v1`)
`{schema, paths[], append_only[], notes[]}`; globs use the kit's matcher (`*` within a segment, `**` across).

### 19.3 EXPECTED.json
`{fixture_id: {commands: [[argv...]], must_fire[], must_not_fire[], must_warn[], expected_exit[], expect: {json_pointer: value | {contains: str}}, source_incident, fm_ids[]}}`.
"Fired" means status FAIL or MISSING. A bare `/pointer` in `expect` resolves in the JSON artifacts the commands
write (gate and canary reports excluded) and, for a command that writes no JSON file (e.g. `kb score` without
`--out`), in its `--json` stdout object. `selftest` runs the fixtures in forked worker processes
(`MINIFLY_SELFTEST_JOBS`, default min(4, CPUs)); the SPEC 12.2 canary step uses C00a's own
`canary <delivery> --seed c0ffee` command. Under `--mutate`, a clean fixture also notices an always-FAIL
replacement of an advisory check (whose FAIL is reported as WARN) when that id newly WARNs there. Example:
`{"C19": {"commands": [["gate", "--stage", "publish", "minifly_corpus_base_dev_v1_20261101"]], "must_fire": ["RUN-01"],
"must_not_fire": [], "must_warn": [], "expected_exit": [1], "expect": {}, "source_incident": "I-0019", "fm_ids": ["FM-19"]}}`.

### 19.4 Gate report (`minifly.gate_report.v1`), checks/GATE_<stage>_<utc>.json or checks/DRY_<stage>_<utc>.json

| Field | Type | Notes |
|---|---|---|
| schema, stage, dry, delivery, utc | | delivery is repo-relative |
| checker_sha256 | str | `common.checker_sha256()`: sha256 over `"<sha256(file)>  <kit-relative path>\n"` lines of every kit file (sorted, `__pycache__`/`*.pyc` excluded); `common.checker_sha256_at(git, commit)` recomputes it from history (CLM-14) |
| checker_version | str | |
| git_mode | str | `real` \| `timeline` (corpus/selftest only; real-mode consumers ignore timeline reports) |
| git_head | str | |
| endpoint_lock_sha256, source_lock_sha256 | str \| null | |
| tier, tier_reasons[] | | computed (Section 4) |
| family_id, family_members[] | | |
| advisory_basis | obj | `{enforce_all, first_source_lock_checker_version}` |
| invoked_by | str | `gate` \| `launch` |
| runtime_s, counts | | measured runtimes for prune reviews |
| results[] | obj | `{id, status, detail, evidence[], data?}`; data carries `runtime_s` |
| exit_code | int | 0..3 |
| verdict | str | last key: `PASS` \| `FAIL` \| `MISSING` \| `CHECKER_ERROR` |

```json
{"schema": "minifly.gate_report.v1", "stage": "roster", "dry": false, "delivery": "minifly_example_dev_v1_20261101",
 "utc": "20261102T101458Z", "checker_sha256": "e695...", "checker_version": "1.0.0", "git_mode": "real", "git_head": "9f3c...",
 "endpoint_lock_sha256": "3333...", "source_lock_sha256": "4444...", "tier": 1, "tier_reasons": ["declared: 1", "..."],
 "family_id": "FAM-1a2b3c4d", "family_members": ["minifly_example_dev_v1_20261101"],
 "advisory_basis": {"enforce_all": false, "first_source_lock_checker_version": "1.0.0"}, "invoked_by": "launch",
 "runtime_s": 2.41, "counts": {"PASS": 40, "NOT_APPLICABLE": 9},
 "results": [{"id": "SELF-01", "status": "PASS", "detail": "checker e695c727f045 clean at HEAD; ...", "evidence": [], "data": {"runtime_s": 0.21}}],
 "exit_code": 0, "verdict": "PASS"}
```

No gate report is written when SELF-01 is in the bundle and did not PASS (a locally modified or
unverified checker never produces a report), and none is written into a non-v1 directory.

Statuses: PASS, FAIL, MISSING, WARN, ADVISORY_FAIL, NOT_APPLICABLE, CHECKER_ERROR. Exit code: 3 for any
CHECKER_ERROR or a non-PASS SELF-01 (or no results); else 1 for any FAIL; else 2 for any MISSING; else 0.
A report counts (CYC-02) only when its verdict and exit_code agree with its own result rows and it names
this delivery. Gate reports are tamper-evident only.

### 19.5 TimelineGit fixture format (corpus and selftest; `--git timeline:<fixture_dir>`)
- `timeline.json` = `{head, commits: [{sha, utc, author, trailers: {key: [values]}, paths: [repo paths touched], branches: [remote branches containing the commit]}], dirty?: [repo paths modified or untracked in the worktree], shallow?: bool}`. History is linear: `is_ancestor(a, b)` iff index(a) <= index(b). Unique sha prefixes of 4+ characters resolve.
- `repo/` holds the final tree (the worktree). Files under `repo/` that no commit touched are untracked.
- `history/<sha>/<path>` holds the content of a path at the commit that last touched it, whenever that differs from `repo/`.
- `file_at(c, p)` uses the latest commit at or before c that touched p: `history/<that sha>/p` if present, else `repo/p` if that commit is the final toucher of p, else None (deleted).
- A `history/<final toucher>/p` that differs from `repo/p` means the worktree is modified (dirty).
- `first_commit_adding(glob)`: first commit touching a matching path that did not exist before it.
- `--git timeline:<fixture_dir>` requires the repo root to lie inside the fixture dir (default `<fixture_dir>/repo`); `launch` refuses timeline mode. In timeline mode SELF-01 checks only the fixture's own `tools/minifly_check` paths (so a fixture can declare `dirty: ["tools/minifly_check/x.py"]` to make it fire); the outer selftest verifies the real kit.
- In timeline mode a fixture commit whose tree holds no `tools/minifly_check` has the empty-kit checker digest
  `sha256(b'')` (CLM-14, committed gate and canary reports); in a real repository such a commit never counts.
- Files beside `repo/` (for example `owner.key` of the held-out mechanism fixtures S44/S45) lie outside the
  fixture's repository; commands may name them with relative paths such as `../owner.key`.
- A governance fixture may carry nested fixtures under `repo/tools/minifly_check/corpus/<id>/` with a nested
  `repo/tools/minifly_check/EXPECTED.json`; GOV-MONO runs the base and head checkers (the fixture's kit files
  overlaid on the running kit) on them (C00f).

```json
{"head": "c3c3...", "dirty": [], "commits": [
  {"sha": "c1c1...", "utc": "20261101T000000Z", "author": "Pencilbard", "trailers": {"Acting-Agent": ["codex"]},
   "paths": ["minifly_corpus_base_dev_v1_20261101/ENDPOINT_LOCK.json"], "branches": ["origin/codex/minifly-corpus"]},
  {"sha": "c3c3...", "utc": "20261102T000000Z", "author": "Pencilbard", "trailers": {"Acting-Agent": ["claude:s1"]},
   "paths": ["minifly_corpus_base_dev_v1_20261101/review/REVIEW.json"], "branches": ["origin/claude/review-x"]}]}
```

### 19.6 Selftest cache
`<tempfile.gettempdir()>/minifly_check_selftest_<checker_sha256>.ok` = `{checker_sha256, checker_version, utc, summary}`,
written by `governance.cmd_selftest` through `common.write_selftest_cache()` after a passing run; SELF-01 accepts it only
when owned by the current user, not world-writable and naming the same checker sha.

### 19.7 RETRO_ANNOTATION rows (retro, read-only)
`{checker_sha256, delivery, check_id, status: PASS | FLAG | NOT_APPLICABLE, reason, evidence_paths[]}`, written only
under `--out` (outside every delivery dir) and later committed to KB/RETRO as annotations, never verdicts.
