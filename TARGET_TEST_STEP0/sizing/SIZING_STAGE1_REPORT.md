# Target-test sizing sweep, stage 1: implementation and qualification (report)

Brief `OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md`, revision 3, from `ea4b2d03` on `codex/minifly-consolidated-20261023`. Released for stage 1 only by the coordinator (review 9.109) and dispatched by the owner's task message of 12 November 2026. Branch `claude/target-test-sizing-stage-1-drmnlq`.

**This is a sizing exercise.** No prediction is judged. Nothing here concludes anything about any learner or rule. **No roster fit was run**: no learner was trained on world 101, 102 or 103, and seeds 1, 2 and 3 were used only to reproduce step 0's record. Every learner in stage 1 was trained on world 104 or on a fixture stream, and then discarded. Stage 2 (the 84 fits) is NOT_RUN and waits for the coordinator's release.

Record: `SIZING_STAGE1_RECORD.json` (rebuilt by `python3 sz_stage1.py core` then `python3 sz_stage1.py record`). Host: 4 cores, Python 3.13, torch 2.15.0+cpu (the same torch as E20-B's receipts), one thread per process, at most three processes at a time.

## First: what failed, and what I changed after a failure

1. **The dependency across a cut was not shown at the distance I registered. Item 5 fails.**
   - *The registered fixture.* The stream is `K <filler> = K \n`, with the key letter repeated exactly D = 60 bytes later. It ran through the arm-B training path for 3,000 updates. The criterion was written into `sz_crosscut.py` before the run: in the last quarter, at least 50% right on at least 100 crossing instances. **Result: 24 of 726 crossing instances right (3.3%) and 102 of 2,371 non-crossing right (4.3%), against a chance rate of 4.2%.** The learner did not learn the 60-byte copy at all at this dose, whether or not a cut lay between key and repeat.
   - *What I ran next.* These are diagnostics, run after the failure. They have no criterion and do not replace the registered fixture.

     | D (bytes) | Updates | Crossing instances right, last quarter | Not crossing, right | Share of instances that cross |
     |---|---|---|---|---|
     | 4 (sanity check of the reading) | 300 | 50 / 50 | 3,150 / 3,150 | 1.6% |
     | **12** | 3,000 | **641 / 643 (99.7%)** | 13,011 / 13,072 (99.5%) | 4.7% |
     | 24 | 3,000 | 11 / 693 (1.6%) | 257 / 6,692 (3.8%) | 9.4% |
     | 40 | 3,000 | 18 / 715 (2.5%) | 154 / 3,856 (4.0%) | 15.6% |
     | 60 (registered) | 3,000 | 24 / 726 (3.3%) | 102 / 2,371 (4.3%) | 23.4% |
     | 60 | 10,000 (about the dose of the largest arm-B fit) | 43 / 2,419 (1.8%) | 340 / 7,904 (4.3%) | 23.4% |
     | 600 (every instance crosses 2 or 3 cuts) | 3,000 | 14 / 319 (4.4%) | none | 100% |

   - *What this says, and what it does not.* At 12 bytes this learner, on its own training path, learns a copy that crosses a cut. The state carries the key across the detach. At 24 bytes and more, within 3,000 updates (and at 60 bytes within 10,000), it learns nothing, crossing or not. Only a copy over random filler was tested, at one model seed. **A dependency across a cut is therefore learnable by this learner at 12 bytes, which meets the brief's wording. It is not learnable at 60 bytes, which is the distance I registered.** I do not count item 5 as passed. Whether D = 12 is enough is the coordinator's call.
   - *Distances the lines actually need* (world 104, aligned lines, from `sz_score.references` with the learner's own chunk grid). For first-encounter probes at modulus 97, the line that states the context value stands at a median of **734 bytes** before the probe (10th to 90th percentile: 100 to 2,811 bytes). The line that states the cue value stands at a median of **411 bytes** (62 to 1,473). Between them and the probe lie a median of 6 and 4 cuts. The figures at 256 presentations are similar (median 783 and 371 bytes). These distances are far beyond the 24 bytes at which the fixture already fails. Under the brief's own rule, **a null on the lines in arm B could not be read** as a statement about the sizes.

2. **Probe fixture F2 failed as I first wrote it, so I widened its sample.** The brief asks to show that a chunk ending with `=` as input changes the response. I first tested this at a single probe, the first one, 46 updates in. With the broken boundary, the logits moved with the revealed digit (largest change 0.064 in arm B, 0.027 in arm A). The three-byte response did not change for any of the ten revealed digits. My first-written condition required a change in bytes, so it failed. The failed attempt is kept in the record (`first_probe_only_criterion_as_first_written: held = false`).

   The extension applies the broken boundary to all 32 acquisition first-encounter probes, with the revealed first digit set to each of 0–9 in turn:
   - The response bytes moved with the reveal at **7 of 32 probes in arm B and 4 of 32 in arm A**.
   - Every probe was pulled toward the revealed digit: the mean gain in the log-probability of the revealed digit was +0.084 in arm B (minimum +0.028) and +0.019 in arm A (minimum +0.016). With no leak this gain is 0.
   - Under the correct boundary, the logits were bitwise identical across all ten reveals.
3. **One modulus fixture was too weak in its first version, so I changed its values.** In the subtraction case, the elimination the solver performs did not pass the modulus. At modulus 11, a solver deliberately left at 97 therefore still gave the right answer from episodes. It was caught only from the lines, and the checker's solver was not caught at all. I changed the fixture values so that the unreduced elimination sum is 2M − 3 for subtraction (and M + 2 for addition). Since then, every component left at another modulus is caught by its answers alone, at all three moduli.

## Admission record

| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Parameterized copies reproduce `TARGET_STEP0_CHECK.json`, seeds 1–3 | **PASS** | All 3,815 values on file compared one by one, including every hash: **0 differences**. All 11 checks pass, each with its fixture failing as required. The default configuration equals step 0's `SIZES`. Only one renaming was allowed: check 2 names the files it read (`sz_oracle`/`sz_solver`) |
| 2 | Modulus fixtures, addition and subtraction at 97, 31, 11 | **PASS** | Wraparound pair right everywhere: 2, and 94 / 28 / 8. Components checked: generator (outcome and written reveal), oracle, solver (from episodes by elimination, from lines, from a prompt), the checker's wiring, and the scorer's direct-use arithmetic. Each component left at another modulus (97→31, 31→11, 11→97) is refused, by configuration hash and by its wrong answers. Each component reports the hash of the configuration it received |
| 3 | Frozen taught panels | **PASS** | `TAUGHT_PANELS.json`: worlds 101, 102, 103 and 104 × moduli 97, 31, 11. Each panel holds 96 pairs per family that the revision does not change, rendered with long filler (10–14 bytes, context outside the last ten) and short filler (0–2). The taught set is identical at 4, 32 and 256 presentations. The file is rebuilt and reproduced exactly |
| — | Scoring code: the brief's admission fixtures | **PASS** | World 104, all 9 settings × 4 paths. The oracle responder scores every first-encounter probe (96/96), both members of every twin (900/900) and every taught item (768/768). The ten-byte-suffix responder scores no twin. In the streams with lines, the direct-use benchmark scores 32/32 acquisition probes and 300/300 twins per panel at the proposed setting |
| 4 | Probe protocol on the real GRU and training path | **PASS** (F2 after its extension, above) | **F1**: the reveal and all later bytes changed (29,831 bytes in arm B; 2,666 episodes in arm A): the response is unchanged and the logits are bitwise equal, for the first first-encounter probe (both arms) and for all 984 prompts of panel A (arm B). **F3**: the second twin member's logits are bitwise equal whether or not the first member came before (24/24); a broken protocol that reads it from the first member's end state changes it (24/24); batched and one-by-one answers agree on 984/984 prompts; the panel leaves the state unchanged. **F4**: probe-on and probe-off runs through panel A and ten more chunks or batches (33 events, 1,016 prompts answered) are bitwise equal in every weight, every AdamW tensor, the carried state, cursor, every counter, the cut list and the loss log, in both arms. A leaky probe differs in 40 entries |
| 5 | Dependency across a cut (D = 60 criterion) | **FAIL** | See above |
| — | Step 0's learner-free checks at all 9 settings | **PASS** | World 104: all 11 checks pass at every modulus and presentation count. Check 8's "another seed" fixture used seed 105, built for that comparison only |
| — | Learner is E20-B's G1 | **PASS** | `sz_verify_shape.py`: same names, shapes and order, and **bitwise-equal initial weights** to `arch20.new_net(seed, "G1")` for model seeds 1101–1104. 1,124,113 parameters; padding row 257; 257 outputs. The sweep's own code does not import arch20 |
| 6 | Forecast within 12 CPU-hours | **PASS** | 4.79 CPU-hours (below) |

**Overall: not admitted as a whole. Item 5 fails as registered, and so `admitted_for_stage_2` is false in the record.** Every other item passes. Whether to release the roster as it stands, to release it without reading the lines in arm B, or to change the fixture or the credit length first is the coordinator's decision. I changed nothing in the design.

## Forecast for the 84 fits

The sizes come from world 104's own streams at every setting. Roster worlds differ only in filler bytes, so their byte counts differ by well under 1%. CPU per update, per probe and per panel comes from the four profiled paths at 256 presentations (500 updates each, run beside at most two other processes).

| Path | Fits | Bytes | Target positions | Updates | CPU training (s) | CPU scoring (s) | CPU generation (s) |
|---|---|---|---|---|---|---|---|
| A, ordinary | 27 | 15,404,904 | 15,404,904 | 21,096 | 1,958 | 29 | 9 |
| B, ordinary | 27 | 15,404,904 | 15,404,904 | 62,829 | 5,474 | 39 | 8 |
| B, lines | 27 | 27,250,344 | 27,250,344 | 109,098 | 9,278 | 40 | 12 |
| B, matched extra bytes (97/4) | 3 | 247,140 | 247,140 | 1,257 | 110 | 3 | 0 |
| **All 84** | 84 | **58,307,292** | **58,307,292** | **194,280** | **16,820** | **110** | **29** |

- **Learner-free checks:** 27 setting-world pairs, about 270 CPU-s in total (about 1.2 s each at 4 presentations, 3 s at 32, 21–24 s at 256).
- **Total: 17,230 CPU-s = 4.79 CPU-hours**, about 1.6 hours of wall time at three processes.
- **Largest fit:** arm B with lines at 97/256: 2,584,950 bytes, 10,196 updates, about 870 CPU-s.

Measured cost per update: 0.0928 s (arm A, a batch of 64 episodes), 0.0871 s (B ordinary), 0.0850 s (B lines), 0.0877 s (B extra). In arm A, target positions are counted as the episodes' bytes (each item is BOS followed by the episode).

**Sizes per setting** (world 104; bytes are the same at every modulus):

| Presentations | Episodes | Ordinary stream: bytes / updates in B / batches in A | With lines or extra bytes: bytes / updates in B |
|---|---|---|---|
| 4 | 3,552 | 46,860 / 281 / 152 | 82,380 / 419 |
| 32 | 15,648 | 204,006 / 895 / 339 | 360,486 / 1,507 |
| 256 | 112,416 | 1,460,790 / 5,805 / 1,853 | 2,584,950 / 10,196 |

Probe cuts add 94–98 updates to each schedule; they are included above.

## Stage 1's own cost

**3,007 CPU-s (0.84 CPU-hours)**, under the two-hour bound. This includes the probe and cross-cut fixtures, the diagnostics, the profile, the checks at all nine settings, and every development and test run. The development runs are itemized in `runs/dev_runs_cpu.json`; a few short ones were not timed and are estimates, marked as such.

## Choices where the brief left room (for the coordinator and the audit)

1. **Where to cut before a probe (arm B).** I cut the chunk exactly at the probe's first answer byte, so the chunk before it has `=` as its last target and the probe reads only the pending `=`. A side effect: each first-encounter reveal starts a new chunk, so its answer digits' gradient cannot reach the context and cue of its own episode (96 episodes per stream). The state still carries them. The alternative also meets the rule's wording: keep the 256-byte grid, read forward without an update from the last grid point to `=`, and replay that chunk afterwards. It avoids the side effect and leaves training identical to a run without probes. Changing to it touches `sz_score.plan` only.
2. **Arm A** cuts its batches at the same probe points (a batch ends before a probe episode). Its grid is otherwise 64 episodes.
3. **Panels scored:** twins at A, I and R (step 0's three places); the taught panel at A and R (the brief's "first panel and end of stream").
4. **"Outcome bytes" for the loss curves** are the two answer digits of every episode. The end byte counts as "other".
5. **A distinct twin pair counts as right** when both members are right on at least 80% of its renderings (step 0's `distinct_pair_share`). "Held success" needs at least 20% of the 24 distinct pairs of both families pooled, so at least 5. "Best recitation shortcut" is the largest of the three tables' accuracies on the same 32 acquisition probes. Both are fixed in `sz_score.classify`.
6. **Tables in arm A** are built from the episodes alone. The ten-byte key is clipped at the episode's start, so short teaching episodes give keys that contain the context byte. In arm B the keys are the stream's last ten bytes.
7. **Cuts between a line and its use** count every chunk boundary after the line's last byte, up to and including the probe's own boundary.
8. **BOS is 257**, E20-B's padding row, so that initial weights equal E20-B's. Output 256 (E20-B's end-of-sequence) is never a target. The greedy response is argmax over all 257 outputs.
9. **Speed changes in the copies, output identical:** the solver splits its buffer once per call, the generator inserts first encounters in one pass, and the disclosure ledger accumulates in one sweep. Parity at the default configuration confirms they give the same results.

## Files (all new, under `TARGET_TEST_STEP0/sizing/`; step 0's four modules untouched)

- **Copies of step 0's modules:**
  - `sz_stream.py`, `sz_oracle.py`, `sz_solver.py`, `sz_check.py`: the parameterized copies. `python3 sz_check.py parity` writes `SZ_PARITY_CHECK.json`.
  - `sz_config.py`: the default configuration (step 0's `SIZES`) and the grid. Only the drivers import it.
- **Fixtures and the frozen panel:**
  - `sz_modfix.py`: the modulus fixtures and the configuration-reach admission.
  - `sz_panels.py`, which writes `TAUGHT_PANELS.json`: the frozen taught panels.
  - `sz_probe_fixtures.py`: probe fixtures F1–F4. `sz_crosscut.py`: the cross-cut fixture.
  - `sz_verify_shape.py`: the one-off comparison with E20-B's network.
- **The sweep itself:**
  - `sz_learner.py`: the GRU, arm A and arm B, and the probe protocol.
  - `sz_score.py`: the schedule, references, readings, the five classes and the fixture responders.
- **Stage 2 driver (not run on any roster world):** `sz_fit.py` (`checks` and `fit`). It refuses a setting that was not admitted, a stream whose hash differs, and an existing output. It was dry-run on world 104 only (`runs/dry_world104/`). Those files show the driver works end to end; they are not fits and are not to be read as results.
- **Profile and record:** `sz_profile.py` writes `runs/profile_*.json`. `sz_stage1.py` writes `runs/stage1_core.json` and `SIZING_STAGE1_RECORD.json`. `runs/` holds every process's output and log.

## Not claimed

- No size decision, no class for any fit, no statement about what this GRU or recipe can acquire from the target stream.
- The cross-cut diagnostics cover one model seed, one fixture kind (a copy over random filler) and doses up to 10,000 updates. They do not say that longer dependencies are unlearnable by this learner under any recipe.
- No formal checker gate, no official seal and no external review is claimed. This record awaits the coordinator and Codex.
