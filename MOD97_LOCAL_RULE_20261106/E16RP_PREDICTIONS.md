# E16-RP predictions: the held-panel seed collision, measured directly (9 November 2026, written before launch)

## Source and setup

- **Brief:** `OPUS_TASK_E16F5_E16RP_CLOUD_JOBS_20261109.md`, Job 2 (commit 923b64a0).
- **Harness:** `exp16_integrated.py` at commit 432e90fb. That includes f38281c, which re-keys the held and within composition panels by split and pair and draws them operand-disjoint from every taught triple. Training triples keep the original seed. No harness file is changed by this job.
- **Work branch:** `claude/vibrant-gauss-tops5f`, reset to `origin/codex/minifly-consolidated-20261023` at 432e90fb. That commit already contains this session's E16-RC work, merged as c8f55e1.

## Comparators

E16-R3's receipts were renamed with `git mv` before any life started:

| Seed | Old name | New name |
|---|---|---|
| 73129 | `exp45/e16_life_T_raw65536_73129.json` | `exp45/e16_life_T_raw65536_73129_oldpanels.json` |
| 93683 | `exp45/e16_life_T_raw65536_93683.json` | `exp45/e16_life_T_raw65536_93683_oldpanels.json` |
| 116089 | `exp45/e16_life_T_raw65536_116089.json` | `exp45/e16_life_T_raw65536_116089_oldpanels.json` |

## Runs

Three lives, detached with `setsid nohup`, launched together. Each is `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>` for one of seeds 73129, 93683, 116089. Each writes `exp45/e16_life_T_raw65536_<seed>.json` and `.pt`, and logs to `exp45/logs/e16rp_<seed>.log`.

## Birth hashes

SHA-256 of `exp45/e11_dev_M_<seed>.pt`, computed before launch:

| Seed | SHA-256 |
|---|---|
| 73129 | `f21c7fb915ede96aaf6d67f2444ec0307e3f6facb2c39a1b4f8a0490c4017ca4` |
| 93683 | `23074e6ddca126d8cce03087dfa2e238c189d7d4e6a78c47ed0a2dceb1c45629` |
| 116089 | `1da65bbdf6b2890f7006fbf45b6035ce7cfe1edcb9c9d2132a020ec7378d9dc9` |

## Predictions (from the brief; not to be changed)

- **P16RP.1 (parity)**
  - Training is identical to E16-R3: same birth, stream, reservoir and draws.
  - So every operation panel equals the old receipt's exactly, at every checkpoint and every k. The panels are `add97`, `sub97`, `max97`, `min97`, `add73`, `sub73`, `max73`, `min73` and the mod-89 ones.
  - The receipt's `settings.birth_sha256` equals the SHA-256 of `exp45/e11_dev_M_<seed>.pt`.
  - Any difference means the run is not the same, and the comparison is void.
- **P16RP.2 (the effect)**
  - On the +- pair, the new panel's final-answer accuracy is within 5 points of the old at every checkpoint from block 4 on.
  - The mean absolute difference across all four held pairs and those checkpoints is under 3 points (panel sampling).
  - If the +- difference exceeds 5 points in a consistent direction, the collision had an effect, and the record's E16-family composition figures carry that correction.

## How these are scored

- **Parity scope.**
  - Parity is checked on the `exact` count of every operation panel at every k present in both receipts, at all seven checkpoints (0–6).
  - It is also checked on the `silenced_k12` operation scores.
  - The training-loss and `reservoir_*` fields are compared as an extra check; they are reported but are not part of the prediction.
- **Effect scope.**
  - "Checkpoints from block 4 on" means stages 4, 5 and 6.
  - Accuracy is `final / n` in percent.
  - "Consistent direction" means the same sign at all three checkpoints, in at least two seeds.

## Pace forecast

- **Basis.** On this four-core container, nine raw lives at once took about 4,700 CPU-s each (E16-RC).
- **Three lives at once.** One core each, with no oversubscription. So about 4,700 CPU-s, or less with less contention.
- **Wall time:** about 1.3 hours.
