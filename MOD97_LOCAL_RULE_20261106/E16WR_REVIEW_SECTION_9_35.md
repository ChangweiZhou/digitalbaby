# 9.35 — E16-WR: bounded stage reservoirs and replay targets



|Prediction|Passes|Total|
|---|---|---|
|P1|0|3|
|P2|0|3|
|P3|1|3|
|P5|3|3|
|P4|10|12|

At self-target capacity16,384, final held composition is 32.08 / 24.88 / 31.58%; observed-target finals are 62.46 / 57.00 / 65.42%. Larger self-target reservoirs are not a monotonic retention repair in these runs. The observed-target comparison must be read through the matched input/coverage checks and its per-seed loss differences, not as a new architecture result.

|Seed|Self loss|Observed loss|Loss reduction pp|P5≥10pp|
|---|---|---|---|---|
|73129|48.9167|16.8333|32.0833|True|
|93683|55.6250|14.0417|41.5833|True|
|116089|52.7917|19.7083|33.0833|True|

All three largest-capacity pairs match actual prompt-order hashes and all nine stage reservoir and draw snapshots exactly. Observed targets were saved with presented examples. This identifies the effect of replacing self-generated replay targets with observed targets at this capacity and schedule, including downstream changes in Store and earlier acquisition. It does not isolate a late-composition-only effect, establish a global capacity optimum, or prove autonomous learning.

Campaign child CPU: 4.4383 hours, including the technical cycles and full default reproduction; controller CPU 1.332s. Campaign wall 46.66min from controller start, excluding prior waiting. Actual peakRSS is recorded per worker in E16WR_FINAL_COSTS.json. Initial forecasts are retained as forecasts, not rewritten to match measured cost.

No final scientific model/optimizer checkpoints were saved under the locked settings. Exact source, hashes, commands, raw receipts, scores, diagnostics and technical trajectory hashes are retained; do not describe these as a complete resumable checkpoint bundle.

Source: E16WR_REPORT.md, E16WR_VERIFICATION.json and exact JOB-selected receipts. Twelve lives, three seeds in each of four arms; one reused-default full reproduction plus separate technical cycles. Existing E16-R review sections were not edited.
