# E12 predictions, frozen before any E12 gradient step

Authorization: user requested execution of Claude's E12 brief at consolidated branch commit 3ce18b84 (7 November 2026). No new mechanisms/tasks. Working model: E8 window18, hidden512, 470,801 parameters; identical targets, splits, optimizer and sealed panels. These are already-opened E8 development panels, not new confirmation data.

A: At reminder256, erased mean >=60%, birth <=20%. If erased is within10 points of birth at all three reminder doses, the brief predicts real erasure. Savings only establishes residual learning, not a unique output-layer mechanism; no savings cannot prove all information absent. From birth, pre-composition and erased WEIGHTS, each reminder starts a fresh matched AdamW; doses64/256/1024 are independent starts with identical batch prefixes across origins and doses. No memory, no replay. Three seeds.
B: At fixed393216 presentations per operation and524288 per composition stage, batch16 retains within3 points of batch64 and improves composition; batch256 loses >10 points on the four add/sub operations. If batch16 composition also within3 points, report no material window effect at this scale. This intervention also changes optimizer update count and cumulative AdamW decay; do not claim pure biological error accumulation.
C: Recency weights0.5^(current-stage-origin-stage), sampled sequentially without replacement from the same eligible prompt buffer. Shares1/2 and1/4. Earliestfour mean loses>10 points vs uniform, latestfour gains<5. Also show all per-operation deltas and whether all are within5. Compare stage8 and final, use stage8 as primary before composition.
D: Buffer128 loses>10 points vs1024; buffer8192 is within3. Primary retention at stage8, also final. Buffer capacity is capped by taught-set size; no repeated entries are manufactured. This tests buffer coverage, not the same quantity as Raviv's repeated exposure count.

Seeds:73129,93683,116089. Sources/checkpoints and reused baselines hashed in E12_LOCK.json. All outputs use e12 prefixes, preserving E9 receipts. No tuning on failures; stop affected part on technical/parity failure; scientific negative predictions remain valid results and are fully reported.

Exact campaign command: ../.venv/bin/python -u e12_campaign.py
The campaign records every exact subprocess command and environment in the lock before launch. Default profile first: E8_RUN_PREFIX=e12_profile_ python exp8_life.py C 73129. Compare complete receipt to exp45/e9_archived_C_73129.json excluding cpu timing only, then record measured forecast before sweep launch. Baseline64/uniform1024 and quarter-uniform reused from E8/E9, not new runs.

A per seed: E8_RUN_PREFIX=e12_A_ E8_SAVE_STAGES=1 E8_SCHEDULE=interleaved E8_REPLAY_SHARE=0 python exp8_life.py A S; then python exp12_reminder.py S ORIGIN R for ORIGIN=erased,birth,pre and R=64,256,1024.
B per seed/batch16,256: E8_RUN_PREFIX=e12_B_ E8_BATCH=B python exp8_life.py C S.
C per seed/share1/2,1/4: E8_RUN_PREFIX=e12_C_ E8_REPLAY_SAMPLING=recency E8_REPLAY_SHARE=SHARE python exp8_life.py C S.
D per seed/cap128,8192: E8_RUN_PREFIX=e12_D_ E8_BUFFER_CAP=CAP python exp8_life.py C S.

Resource plan:21 new full lives +1 full parity/profile life,27 short independent reminders,9 saved stage checkpoints (model, optimizer and RNG, ~51MB before serialization overhead);3 existing birth checkpoints reused (~5.7MB). No development training. Model weight tensors1,883,204bytes, Adam moments3,766,408bytes per worker, plus activations/store/buffer; peak RSS measured per subprocess where supported. At most6 single-threaded subprocesses. Forecast pending the first profile; no extrapolation is a measured cost.

Path-integration follow-up remains NOT_RUN, source brief recorded separately in E12_LOCK.json; E12 does not complete it. E11 belongs to Claude and is not modified.
