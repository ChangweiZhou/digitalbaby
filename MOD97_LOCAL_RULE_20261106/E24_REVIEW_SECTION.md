# E24: every-byte targets at birth, the loss or the rule

E24 grew the windowless two-bank learner of record (one head at the top) from nothing. Each arm ran 24,576 updates on seeds 73129, 93683 and 116089, all on one cloud host class. The arms differ in two things:
- the target: answer bytes only, or every next byte;
- the training configuration: backprop with decay 0.30, or the sign sweep with decay 0.25.

The measure is the hard mean, add89 and sub89 at k = 12, whole answers.

| Arm | Hard mean (73129 / 93683 / 116089) |
|---|---|
| Backprop, answer bytes (baseline, born on this host class) | 93.2 / 92.8 / 92.8 |
| Backprop, every byte | **91.5 / 89.3 / 91.2** |
| Sign sweep, answer bytes (cloud-born originals) | 87.5 / 90.7 / 84.7 |
| Sign sweep, every byte | **0.8 / 1.0 / 1.2** |
| Sign sweep, every byte, answer bytes half the loss | **81.5 / 79.8 / 83.3** |

**All three predictions hold, in all three seeds.**
- P24.1: backprop under every-byte targets stays at least 60.
- P24.2: the sign sweep under every-byte targets ends at most 10.
- P24.3: the balanced sign sweep reaches at least 40.

**What it shows.**
- **Every-byte targets cost backprop 1.7–3.5 points and the sign sweep 83.5–89.7.**
  - That is an interaction between objective and configuration of 81.8–86.2 points, in this network at this dose.
  - The configurations also differ in decay, so the update is not isolated.
- **The sign sweep does not fail to start under every-byte targets.**
  - It reaches 8–15 at 8,192 updates and then falls to about 1.
  - Its max/min mean ends below the operand-copy level of 50.
- **Giving the answer bytes half the loss restores the sign sweep in every seed.**
  - The gain over the unweighted arm is 78.8–82.2 points.
  - It ends 1.4–10.9 below the answer-only sign sweep.
  - The weighting reads the answer mask. It is a diagnostic, so it says nothing about where such a weighting could come from without one.

**Host checks.**
- The backprop baseline was rerun on this host for all three seeds, and the runs equal E20-B's G0 births in every sealed score and weight tensor.
- The owner's-machine births on file differ from these by +2.0 / −1.7 / +2.5 points and are not used.
- The sign-sweep original of seed 73129 reproduced bit for bit.

**Run.**
- The nine new births used 30,528 CPU-s and 2 h 52 min, three at a time on a four-core cloud Xeon; the forecast was 30,000–32,000.
- Sources as hashed in the report.
- No interruption, rerun or resume.

**Not run:** a balanced backprop arm, a sign sweep at decay 0.30, lives, filler streams and local heads.

Development result on sealed panels, evidence level E2, births only. Nothing is claimed about lives, composition or the target test. Decisions remain Mr. Zhou's.

Sources: `E24_REPORT.md`, `E24_PREDICTIONS.md`, `E24_HOST_CHECK.md`, `E24_HOST_CHECK_2.md`, `exp45/e24/E24_TABLES.md`, the nine receipts and their logs.
